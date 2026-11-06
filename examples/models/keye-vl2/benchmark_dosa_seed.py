# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay original layer0 inputs with cold representative-query and carry
seeds."""

import argparse
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from benchmark_dosa import layer_zero
from diagnose_dosa import atomic_json
from sweep_dosa_heads import load_before

from relax.models.keye_vl2 import dosa_parallel
from relax.models.keye_vl2.dosa import DoSASeedState, dosa_select
from relax.models.keye_vl2.indexer_scores import dense_indexer_scores
from relax.models.keye_vl2.sparse_attention import compact_indices
from relax.models.keye_vl2.topk import dsa_topk_indices
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
START = time.monotonic()


@torch.no_grad()
def measure_case(data: dict, heads: int, before, output: Path, deadline: float) -> list[dict]:
    q, k, w = (data[name].contiguous() for name in ("iq", "ik", "iw"))
    offsets = data["offsets"]
    n, rows = len(k), len(q)
    causal = torch.arange(n, device=q.device)[None] <= offsets[:, None]
    dense = dense_indexer_scores(q, k, w)
    masked = dense.masked_fill(~causal, torch.finfo(q.dtype).min)
    expected = dsa_topk_indices(masked)
    expected_compact = compact_indices(expected, causal.gather(1, expected.long()))
    full_tau = masked.topk(2048, dim=-1).values[:, -1].float()
    previous_offsets = data["previous_offsets"]
    previous_causal = torch.arange(n, device=q.device)[None] <= previous_offsets[:, None]
    previous_scores = dense_indexer_scores(data["previous_iq"].contiguous(), k, data["previous_iw"].contiguous())
    previous_ids = dsa_topk_indices(previous_scores.masked_fill(~previous_causal, torch.finfo(q.dtype).min))
    history = DoSASeedState()
    history.publish(previous_ids, previous_offsets, n, previous_causal.gather(1, previous_ids.long()))

    def baseline():
        scores = dense_indexer_scores(q, k, w).masked_fill(~causal, torch.finfo(q.dtype).min)
        ids = dsa_topk_indices(scores)
        return compact_indices(ids, causal.gather(1, ids.long()))

    modes = [
        ("uniform4096", 4096, False),
        ("cold4096", 4096, False),
        ("carry4096", 4096, True),
        ("cold12288", 12288, False),
        ("carry12288", 12288, True),
    ]

    def selector(name, size, carry, stats=None):
        # Every repetition consumes the preceding128 queries, never winners
        # from an earlier replay of the current128 queries.
        state = DoSASeedState(history.selected, history.visible_end, history.key_count) if carry else DoSASeedState()
        if name == "uniform4096" and stats is None and before is not None:
            return before(q, k, w, offsets, heads=heads, seed_size=4096)
        return dosa_select(
            q,
            k,
            w,
            offsets,
            heads=heads,
            seed_size=size,
            seed_policy="uniform" if name == "uniform4096" else "informative",
            seed_state=None if name == "uniform4096" else state,
            stats=stats,
        )

    def compact_mode(name, size, carry):
        ids = selector(name, size, carry)
        return baseline() if ids is None else compact_indices(ids.clamp_min(0), ids >= 0)

    records = []
    for name, size, carry in modes:
        if time.monotonic() > deadline - 20:
            raise TimeoutError("shared GPU budget during seed diagnostics")
        stats = {}
        if n <= size:
            assert selector(name, size, carry, stats) is None
            records.append(dict(mode=name, seed_size=size, fallback=True, rows=[]))
            continue
        capacity = n - size
        bounds = torch.empty((rows, capacity), device=q.device)
        original = dosa_parallel.screen

        class Capture:
            def __getitem__(self, grid):
                def call(*args, **kwargs):
                    return original[grid](*args, BOUNDS=bounds, DUMP=True, **kwargs)

                return call

        with patch.object(dosa_parallel, "screen", Capture()):
            ids = selector(name, size, carry, stats)
        for a, b in zip(compact_indices(ids.clamp_min(0), ids >= 0), expected_compact):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        seeds = stats["seed_ids"]
        tau = dense.gather(1, seeds.long()).topk(2048, dim=-1).values[:, -1].float()
        torch.testing.assert_close(tau, stats["seed_threshold"], atol=0, rtol=0)
        if name == "uniform4096":
            t = torch.arange(capacity, device=q.device, dtype=torch.int64)[None]
            base = stats["seed_bases"].long()[:, None]
            missing = base - size
            rank = torch.where(
                t < missing, ((t + 1) * base + missing.clamp_min(1) - 1) // missing.clamp_min(1) - 1, t + size
            )
            candidates = stats["seed_order"][rank]
        else:
            candidates = stats["seed_order"][:, size:]
            torch.testing.assert_close(
                stats["seed_order"].sort().values,
                torch.arange(n, device=q.device, dtype=torch.int32).expand_as(stats["seed_order"]),
            )
            assert int(stats["seed_source"]) == int(carry)
        candidates = candidates.repeat_interleave(4, dim=0)[:rows]
        partition = torch.cat((seeds, candidates), -1).sort().values
        torch.testing.assert_close(partition, torch.arange(n, device=q.device, dtype=torch.int32).expand_as(partition))
        live = candidates <= offsets[:, None]
        score = dense.gather(1, candidates.long()).float()
        assert ((bounds >= score) | ~live).all(), "certificate underestimates original scorer"
        rejected = live & (bounds < tau[:, None])
        score_only = live & (score < tau[:, None])
        full_score_only = live & (score < full_tau[:, None])
        counts = live.sum(-1)
        pruned = rejected.sum(-1)
        exact = score_only.sum(-1)
        torch.testing.assert_close((counts - pruned).int(), stats["survivors"])
        torch.testing.assert_close((2048 + counts - exact).int(), stats["selection_candidates"])
        fields = dict(
            candidates=counts,
            pruned_count=pruned,
            score_only_count=exact,
            full_score_only_count=full_score_only.sum(-1),
            tau_seed=tau,
            tau_full=full_tau,
            seed_count=stats["seed_count"],
            visible=stats["visible"],
            selection_candidates=stats["selection_candidates"],
            exact_score_rejected=stats["exact_score_rejected"],
        )
        cpu = {key: value.cpu().tolist() for key, value in fields.items()}
        per_query = [dict(query=i, **{key: value[i] for key, value in cpu.items()}) for i in range(rows)]
        denominator = int(counts.sum())
        summary = dict(
            candidates=denominator,
            pruned=int(pruned.sum()),
            prune_percent=100 * int(pruned.sum()) / denominator,
            score_only_percent=100 * int(exact.sum()) / denominator,
            full_score_only_percent=100 * int(full_score_only.sum()) / denominator,
            whole_visible_skipped_percent=100 * int(pruned.sum()) / int(stats["visible"].sum()),
            tau_seed_mean=float(tau.mean()),
            tau_full_mean=float(full_tau.mean()),
            upper_slack_mean=float((bounds - score)[live].mean()),
        )
        np.savez_compressed(
            output / f"{data['label']}-n{n}-r{heads}-{name}-mapping.npz",
            seeds=seeds.cpu().numpy(),
            order=stats["seed_order"].cpu().numpy(),
            bases=stats["seed_bases"].cpu().numpy(),
        )
        records.append(dict(mode=name, seed_size=size, fallback=False, rows=per_query, summary=summary))

    functions = [("dsa", baseline)]
    for name, size, carry in modes:
        functions.append((name, lambda name=name, size=size, carry=carry: compact_mode(name, size, carry)))
    for _, function in functions:
        for _ in range(3):
            function()
    torch.cuda.synchronize()
    samples = {name: [] for name, _ in functions}
    for repeat in range(7):
        if time.monotonic() > deadline - 10:
            raise TimeoutError("shared GPU budget during complete selector timing")
        rotated = functions[repeat % len(functions) :] + functions[: repeat % len(functions)]
        for name, function in rotated[:: 1 if repeat % 2 == 0 else -1]:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            function()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end))
    for record in records:
        name = record["mode"]
        record["timing"] = dict(
            dsa_ms=statistics.median(samples["dsa"]),
            selector_ms=statistics.median(samples[name]),
            uniform_ms=statistics.median(samples["uniform4096"]),
            samples_ms=samples[name],
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before-dir", type=Path, help="Optional directory containing before-*.py selector snapshots")
    parser.add_argument("--budget-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.budget_seconds <= 900:
        parser.error("GPU budget must be within [1,900]")
    deadline = START + args.budget_seconds
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    snapshot_dir = args.before_dir or args.output
    has_snapshots = all(
        (snapshot_dir / f"before-{name}.py").is_file() for name in ("dosa", "dosa_kernels", "dosa_parallel")
    )
    before = load_before(snapshot_dir) if args.before_dir is not None or has_snapshots else None
    state = dict(
        cases=[],
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        uniform_timing_reference=str(snapshot_dir) if before is not None else "current_selector",
    )
    try:
        for n in (8192, 16384):
            for entry in manifest["samples"]:
                if entry["tokens"] < n:
                    raise ValueError("input shorter than requested natural prefix")
                all_data = layer_zero(args.checkpoint, Path(entry["text_file"]).read_text(), n, 256)
                data = dict(
                    iq=all_data["iq"][-128:],
                    ik=all_data["ik"],
                    iw=all_data["iw"][-128:],
                    offsets=all_data["offsets"][-128:],
                    previous_iq=all_data["iq"][:-128],
                    previous_iw=all_data["iw"][:-128],
                    previous_offsets=all_data["offsets"][:-128],
                    label=entry["label"],
                )
                for heads in (8, 10, 12):
                    label = f"{entry['label']}-n{n}-r{heads}"
                    if time.monotonic() > deadline - 30:
                        state["cases"].append(dict(case=label, status="not_run_time_budget"))
                        continue
                    results = measure_case(data, heads, before, args.output, deadline)
                    atomic_json(
                        args.output / f"{label}.json",
                        dict(input=entry, layer=0, length=n, heads=heads, results=results),
                    )
                    state["cases"].append(dict(case=label, status="ok"))
                    state["elapsed_seconds"] = time.monotonic() - START
                    atomic_json(args.output / "status.json", state)
                    logger.info(
                        "%s %s",
                        label,
                        [
                            (
                                r["mode"],
                                None if r["fallback"] else round(r["summary"]["prune_percent"], 3),
                                round(r["timing"]["selector_ms"], 3),
                            )
                            for r in results
                        ],
                    )
                del data, all_data
                torch.cuda.empty_cache()
    except Exception as error:
        state["error"] = f"{type(error).__name__}: {error}"
        logger.exception("Seed sweep stopped")
        raise
    finally:
        state["elapsed_seconds"] = time.monotonic() - START
        atomic_json(args.output / "status.json", state)


if __name__ == "__main__":
    main()
