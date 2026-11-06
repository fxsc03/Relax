# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Layer-zero R=4 attribution; scalar thresholds are not selector timings.

Run with a prepared token manifest and an outer timeout of at most 900 seconds.
Only the diagnostic uses torch.topk for scalar thresholds. Production kernels,
BF16 rounding, four-query seeds, and certificate protection are unchanged.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import triton
from benchmark_dosa import layer_zero

from relax.models.keye_vl2 import dosa_kernels as kernels
from relax.models.keye_vl2 import dosa_parallel as parallel
from relax.models.keye_vl2.indexer_scores import dense_indexer_scores
from relax.utils.logging_utils import get_logger


START = time.monotonic()
logger = get_logger(__name__)


def atomic_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def distribution(values: torch.Tensor) -> dict:
    values = values.float().cpu()
    finite = values[torch.isfinite(values)]
    result = {"count": values.numel(), "nonfinite": int((~torch.isfinite(values)).sum())}
    if finite.numel():
        quantiles = torch.quantile(finite, torch.tensor([0.1, 0.5, 0.9, 0.99])).tolist()
        result.update(zip(("p10", "p50", "p90", "p99"), quantiles))
        result.update(min=float(finite.min()), max=float(finite.max()))
    return result


@torch.no_grad()
def diagnose(data: dict, deadline: float, heads: int = 4) -> tuple[list, dict]:
    q, k, w = (data[x].contiguous() for x in ("iq", "ik", "iw"))
    offsets = data["offsets"]
    n, s, r = len(k), 4096, heads
    cap = n - s
    tiles = triton.cdiv(cap, 128)
    valid_ids = torch.arange(n, device="cuda", dtype=torch.int32)
    launch = {"num_warps": 4, "enable_fp_fusion": False}
    records, seed_records, base_records, projection_records = [], [], [], []
    override_ranks = data.get("head_ranks")
    for start in range(0, len(q), 4):
        if time.monotonic() > deadline - 10:
            raise TimeoutError("case incomplete: total GPU deadline")
        a, b = q[start : start + 4].contiguous(), w[start : start + 4].contiguous()
        rows = len(a)
        visible = (offsets[start : start + rows] + 1).int()
        seeds = torch.empty((rows, s), device="cuda", dtype=torch.int32)
        bases = torch.empty((1,), device="cuda", dtype=torch.int32)
        seed_scores = a.new_empty((rows, s))
        plan = torch.empty((rows, r * 66 + 193), device="cuda")
        vectors = a.new_zeros((rows, r + 1, 64))
        coefficients = torch.zeros((rows, r), device="cuda")
        error = torch.full((rows,), float("inf"), device="cuda")
        norms = torch.empty((n,), device="cuda")
        flags = torch.empty((rows, cap), device="cuda", dtype=torch.bool)
        counts = torch.empty((rows, tiles), device="cuda", dtype=torch.int32)
        unsafe = torch.empty_like(counts)
        bounds = torch.empty((rows, cap), device="cuda")
        parallel.make_seeds[(rows, s // 128)](valid_ids, visible, seeds, bases, n, rows, s, **launch)
        kernels.seed_scores[(rows, s // 128)](
            a, k, b, seeds, visible, seed_scores, a.stride(0), b.stride(0), s, **launch
        )
        dense = dense_indexer_scores(a, k, b).float()
        nonfinite = (~torch.isfinite(dense) & (valid_ids[None] < visible[:, None])).sum(-1)
        if bool(nonfinite.any()):
            raise ValueError(f"nonfinite scores per row: {nonfinite.cpu().tolist()}")
        torch.testing.assert_close(seed_scores.float(), dense.gather(1, seeds.long()), atol=0, rtol=0)
        tau = seed_scores.float().topk(2048, dim=-1).values[:, -1].contiguous()
        causal = valid_ids[None] < visible[:, None]
        full = dense.masked_fill(~causal, -float("inf")).topk(2048, dim=-1).values[:, -1]
        projection = torch.empty((rows, 16, 20), device="cuda")
        options = {}
        if override_ranks is not None:
            ranks = override_ranks[start : start + rows].cuda().contiguous()
            options = {"HEAD_RANKS": ranks, "OVERRIDE": True}
        kernels.prepare[(rows,)](
            a,
            b,
            k,
            seeds,
            visible,
            plan,
            a.stride(0),
            b.stride(0),
            s,
            r,
            DIAG=projection,
            DUMP=True,
            **options,
            **launch,
        )
        assert bool(torch.isfinite(projection).all()), "nonfinite projection diagnostic"
        assert bool((projection[:, :, :16] >= 0).all()), "negative projection coefficient"
        for row in range(rows):
            ranked = projection[row, :, 19].long()
            priority = b[row].float() * a[row].float().norm(dim=-1)
            positive = b[row] > 0
            if override_ranks is not None:
                assert torch.equal(ranked, ranks[row].long()), "requested head permutation changed"
            elif int(positive.sum()) >= r:
                expected_heads = priority.masked_fill(~positive, -float("inf"))
                expected_heads = expected_heads.argsort(descending=True, stable=True)[:r]
                assert torch.equal(ranked.argsort()[:r], expected_heads), "retained head selection changed"
            vectors_ref = a[row, ranked.argsort()[:r]].double()
            fit_ref = projection[row, :, :r].double() @ vectors_ref
            omitted = positive & (ranked >= r)
            actual = ((a[row].double() - fit_ref) * b[row, :, None].double()).norm(dim=-1)
            assert bool((projection[row, omitted, 17].double() >= actual[omitted]).all())

        parallel.pack_plan[(rows,)](plan, visible, vectors, coefficients, error, r, s, **launch)
        parallel.key_norm[(triton.cdiv(n, 128),)](k, norms, n, **launch)
        parallel.screen[(1, tiles)](
            k,
            norms,
            vectors,
            coefficients,
            error,
            tau,
            bases,
            valid_ids,
            visible,
            flags,
            counts,
            unsafe,
            rows,
            n,
            s,
            r,
            cap,
            tiles,
            4 * max(4, triton.next_power_of_2(r + 1)),
            BOUNDS=bounds,
            DUMP=True,
            **launch,
        )
        # Exact inverse of production complement(), not original token columns.
        t = torch.arange(cap, device="cuda", dtype=torch.int64)
        base = bases[0].long()
        missing = base - s
        rank = torch.where(t < missing, ((t + 1) * base + missing.clamp_min(1) - 1) // missing.clamp_min(1) - 1, t + s)
        ids = valid_ids[rank.long()]
        for row in range(rows):
            mask = rank < visible[row]
            original = ids[mask].long()
            scores, upper = dense[row, original], bounds[row, mask]
            seed_ids = seeds[row].long()
            assert not bool(torch.isin(original, seed_ids).any()), "seed in complement"
            assert original.numel() == int(visible[row]) - s, "candidate coverage mismatch"
            assert bool((upper >= scores).all()), "certificate below original BF16 score"
            assert bool(tau[row] <= full[row]), "seed threshold exceeds full threshold"
            rejected = upper < tau[row]
            possible = scores < tau[row]
            assert bool((~rejected | possible).all()), "screen rejected non-rejectable score"
            assert int((~rejected).sum()) == int(counts[row].sum()), "live-count mismatch"
            assert int(torch.isinf(upper).sum()) == int(unsafe[row].sum()), "unsafe-count mismatch"
            assert torch.equal(flags[row, mask], ~rejected), "flag mismatch"
            den = scores.numel()
            sample = torch.linspace(0, den - 1, min(16, den), device="cuda").long()
            omitted = (b[row] > 0) & (projection[row, :, 19] >= r)
            original_norm = projection[row, omitted, 18]
            single_norm = projection[row, omitted, 16]
            multi_norm = projection[row, omitted, 17]
            record = dict(
                single_residual_sum=float(single_norm.sum()),
                multi_residual_sum=float(multi_norm.sum()),
                single_residual_ratio=float((single_norm / original_norm).mean()) if single_norm.numel() else None,
                multi_residual_ratio=float((multi_norm / original_norm).mean()) if multi_norm.numel() else None,
                omitted_positive_heads=int(omitted.sum()),
                query=start + row,
                offset=int(offsets[start + row]),
                candidates=den,
                tau_seed=float(tau[row]),
                tau_full=float(full[row]),
                threshold_gap=float(full[row] - tau[row]),
                score_only_count=int(possible.sum()),
                pruned_count=int(rejected.sum()),
                full_score_only_count=int((scores < full[row]).sum()),
                missed_count=int((possible & ~rejected).sum()),
                seed_ties_count=int((scores == tau[row]).sum()),
                full_ties_count=int((scores == full[row]).sum()),
                nonfinite_scores=int(nonfinite[row]),
                infinite_bounds=int(torch.isinf(upper).sum()),
                certificate_bypass=not bool(torch.isfinite(error[row])),
                certificate_error=float(error[row]) if torch.isfinite(error[row]) else None,
                fallback=False,
                positive_heads=int((b[row] > 0).sum()),
                negative_heads=int((b[row] < 0).sum()),
                bound_minus_score=distribution(upper - scores),
                bound_minus_seed=distribution(upper - tau[row]),
                bound_samples=[
                    dict(token_id=int(i), score=float(v), upper=float(u) if torch.isfinite(u) else None)
                    for i, v, u in zip(original[sample].cpu(), scores[sample].cpu(), upper[sample].cpu())
                ],
            )
            for field in ("score_only", "pruned", "full_score_only", "missed", "seed_ties", "full_ties"):
                record[field + "_fraction"] = record[field + "_count"] / den if den else None
            records.append(record)
        from relax.models.keye_vl2.dosa import dosa_select
        from relax.models.keye_vl2.topk import dsa_topk_indices

        selection_stats = {}
        if override_ranks is None:
            selected = dosa_select(
                a,
                k,
                b,
                offsets[start : start + rows],
                heads=r,
                seed_size=s,
                seed_policy="uniform",
                stats=selection_stats,
            )
        else:
            from unittest.mock import patch

            original_prepare = kernels.prepare

            class ForcedPrepare:
                def __getitem__(self, grid):
                    def call(*args, **kwargs):
                        return original_prepare[grid](*args, **options, **kwargs)

                    return call

            with patch.object(kernels, "prepare", ForcedPrepare()):
                selected = dosa_select(
                    a,
                    k,
                    b,
                    offsets[start : start + rows],
                    heads=r,
                    seed_size=s,
                    seed_policy="uniform",
                    stats=selection_stats,
                )
        for index, record in enumerate(records[-rows:]):
            record["selection_candidates"] = int(selection_stats["selection_candidates"][index])
            record["exact_score_rejected"] = int(selection_stats["exact_score_rejected"][index])
        assert selected is not None, "unexpected backend fallback"
        expected = dsa_topk_indices(dense.to(a.dtype).masked_fill(~causal, torch.finfo(a.dtype).min))
        torch.testing.assert_close(selected.sort(-1).values, expected.int().sort(-1).values, atol=0, rtol=0)
        projection_records.append(projection.cpu().numpy())
        seed_records.append(seeds.cpu().numpy())
        base_records.append(bases.cpu().numpy())
    return records, dict(
        projections=np.concatenate(projection_records),
        seeds=np.concatenate(seed_records),
        bases=np.concatenate(base_records),
        valid_ids=valid_ids.cpu().numpy(),
    )


def summarize(records: list) -> dict:
    total = sum(x["candidates"] for x in records)
    result = {"candidates": total, "queries": len(records)}
    for field in ("score_only", "pruned", "full_score_only", "missed", "seed_ties", "full_ties"):
        count = sum(x[field + "_count"] for x in records)
        result[field + "_fraction"] = count / total if total else None
        fractions = torch.tensor([x[field + "_fraction"] for x in records])
        result[field + "_query_distribution"] = distribution(fractions)
    for field in ("tau_seed", "tau_full", "threshold_gap"):
        result[field] = distribution(torch.tensor([x[field] for x in records]))
    for field in ("nonfinite_scores", "infinite_bounds", "certificate_bypass", "fallback"):
        result[field] = sum(x[field] for x in records)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=900)
    parser.add_argument("--head-search-dir", type=Path, help="CPU-validated per-case rank permutations")
    args = parser.parse_args()
    if not 1 <= args.budget_seconds <= 900:
        parser.error("GPU budget must be in [1, 900]")
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = START + args.budget_seconds
    manifest = json.loads(args.manifest.read_text())
    status = {"gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(), "cases": []}
    try:
        from fast_hadamard_transform import hadamard_transform

        x = torch.ones((4, 64), device="cuda", dtype=torch.bfloat16)
        y = hadamard_transform(x, scale=0.125)
        torch.cuda.synchronize()
        assert bool(torch.isfinite(y).all()) and float(y[0, 0]) == 8
        assert torch.cuda.get_device_capability()[0] in (8, 9)
        # The first real 8K case checks BF16 scorer, seed equality, screen mapping,
        # certificate inequalities, and kernel live counts before adding cases.
        for n in (8192, 16384, 65536, 131072):
            for entry in manifest["samples"]:
                name = f"{entry['label']}-n{n}"
                item = dict(case=name, length=n, layer=0, queries=128, status="not_run_time_budget")
                status["cases"].append(item)
                if time.monotonic() > deadline - 30:
                    continue
                # TODO(agent): production complement uses overflowing i32 arithmetic.
                # Keep this diagnostic from consuming invalid bound/token mappings.
                if (n - 4096) * (n + 1) > torch.iinfo(torch.int32).max:
                    item["status"] = "not_run_unsafe_complement_int32"
                    continue
                if entry["tokens"] < n:
                    item["status"] = "input_too_short"
                    continue
                text = Path(entry["text_file"]).read_text()
                data = layer_zero(args.checkpoint, text, n, 128)
                if args.head_search_dir:
                    head_ranks = np.load(args.head_search_dir / f"{name}-ranks.npy")
                    if head_ranks.shape != (128, 16):
                        raise ValueError("head rank shape must be [128, 16]")
                    if not np.array_equal(np.sort(head_ranks, axis=1), np.tile(np.arange(16), (128, 1))):
                        raise ValueError("each head rank row must be a permutation of 0..15")
                    data["head_ranks"] = torch.from_numpy(head_ranks.astype(np.int32))
                rows, audit = diagnose(data, deadline)
                np.savez_compressed(args.output / f"{name}-mapping.npz", **audit)
                atomic_json(
                    args.output / f"{name}.json",
                    dict(input=entry, layer=0, length=n, rows=rows, summary=summarize(rows)),
                )
                item["status"] = "ok"
                logger.info("Completed %s: %s", name, summarize(rows)["pruned_fraction"])
                del data, rows, audit
                torch.cuda.empty_cache()
                status["elapsed_seconds"] = time.monotonic() - START
                atomic_json(args.output / "status.json", status)
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        if status["cases"]:
            status["cases"][-1]["status"] = "error"
        logger.exception("Diagnostic stopped; no automatic retry")
        raise
    finally:
        existing = {x["case"] for x in status["cases"]}
        for n in (8192, 16384, 65536, 131072):
            for entry in manifest["samples"]:
                name = f"{entry['label']}-n{n}"
                if name not in existing:
                    status["cases"].append(
                        dict(case=name, length=n, layer=0, queries=128, status="not_run_preflight_or_case_error")
                    )
        status["elapsed_seconds"] = time.monotonic() - START
        atomic_json(args.output / "status.json", status)


if __name__ == "__main__":
    main()
