# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Measure batch-size amortization over a fixed real query set and frozen
carry."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from benchmark_dosa import layer_zero
from diagnose_dosa import atomic_json

from relax.models.keye_vl2.dosa import DoSASeedState, dosa_select
from relax.models.keye_vl2.indexer_scores import dense_indexer_scores
from relax.models.keye_vl2.sparse_attention import compact_indices
from relax.models.keye_vl2.topk import dsa_topk_indices
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
START = time.monotonic()


@torch.no_grad()
def audit_certificate(data: dict, history: DoSASeedState, heads: int, deadline: float) -> dict:
    """Check every live candidate bound against original scores in bounded
    batches."""
    from unittest.mock import patch

    from relax.models.keye_vl2 import dosa_parallel

    query, key, weights, offsets = (data[name] for name in ("iq", "ik", "iw", "offsets"))
    width, seed_size = len(key), 12288
    checked = 0
    infinite = 0
    minimum_slack = float("inf")
    original_screen = dosa_parallel.screen
    for start in range(0, len(query), 128):
        if time.monotonic() > deadline - 30:
            raise TimeoutError("shared GPU deadline during exhaustive certificate audit")
        stop = min(start + 128, len(query))
        upper = torch.empty((stop - start, width - seed_size), device=query.device)

        class Capture:
            def __init__(self, bounds: torch.Tensor) -> None:
                self.bounds = bounds

            def __getitem__(self, grid):
                def call(*args, **kwargs):
                    return original_screen[grid](*args, BOUNDS=self.bounds, DUMP=True, **kwargs)

                return call

        state = DoSASeedState(history.selected, history.visible_end, history.key_count)
        stats = {}
        with patch.object(dosa_parallel, "screen", Capture(upper)):
            result = dosa_select(
                query[start:stop],
                key,
                weights[start:stop],
                offsets[start:stop],
                heads=heads,
                seed_size=seed_size,
                seed_state=state,
                stats=stats,
            )
        assert result is not None
        assert int(stats["seed_source"]) == 1
        ordering = stats["seed_order"]
        torch.testing.assert_close(
            ordering[0].sort().values, torch.arange(width, device=query.device, dtype=torch.int32)
        )
        assert (ordering == ordering[0, None]).all()
        assert (stats["seed_ids"] == ordering[0, :seed_size][None]).all()
        dense = dense_indexer_scores(query[start:stop], key, weights[start:stop])
        seed_scores = dense.gather(1, stats["seed_ids"].long())
        tau = seed_scores.topk(2048, dim=-1).values[:, -1].float()
        torch.testing.assert_close(tau, stats["seed_threshold"], atol=0, rtol=0)
        candidate_ids = ordering[:, seed_size:].repeat_interleave(4, dim=0)[: stop - start]
        live = candidate_ids <= offsets[start:stop, None]
        scores = dense.gather(1, candidate_ids.long()).float()
        assert (torch.isfinite(scores) | ~live).all(), "nonfinite original score in real input"
        assert ((upper >= scores) | ~live).all(), "certificate underestimates original score"
        rejected = live & (upper < tau[:, None])
        assert not (rejected & (scores >= tau[:, None])).any(), "unsafe threshold rejection"
        expected_survivors = live.sum(-1) - rejected.sum(-1)
        torch.testing.assert_close(expected_survivors.int(), stats["survivors"])
        checked += int(live.sum())
        infinite += int((torch.isinf(upper) & live).sum())
        slack = (upper - scores).masked_fill(~live, float("inf")).amin()
        minimum_slack = min(minimum_slack, float(slack))
        del upper, result, stats, ordering, dense, seed_scores, candidate_ids, live, scores, rejected, state
    return dict(
        checked_candidate_pairs=checked,
        underestimated_pairs=0,
        unsafe_rejections=0,
        infinite_bounds=infinite,
        minimum_bound_slack=minimum_slack if minimum_slack < float("inf") else None,
    )


@torch.no_grad()
def measure_batch(
    data: dict, history: DoSASeedState, expected: tuple, batch_size: int, deadline: float, heads: int = 12
) -> dict:
    query, key, weights, offsets = (data[name] for name in ("iq", "ik", "iw", "offsets"))
    causal = data["causal"]
    total_queries, width = len(query), len(key)
    seed_size = 12288
    target_seed = None
    pruned = torch.zeros((), device=query.device, dtype=torch.int64)
    candidates = torch.zeros_like(pruned)
    survivors = torch.zeros_like(pruned)
    visible = torch.zeros_like(pruned)
    chunks = []
    thresholds = []
    survivor_counts = []
    for start in range(0, total_queries, batch_size):
        stop = min(start + batch_size, total_queries)
        chunks.append((start, stop))
        if time.monotonic() > deadline - 20:
            raise TimeoutError("shared GPU deadline during scale correctness checks")
        frozen = DoSASeedState(history.selected, history.visible_end, history.key_count)
        stats = {}
        selected = dosa_select(
            query[start:stop],
            key,
            weights[start:stop],
            offsets[start:stop],
            heads=heads,
            seed_size=seed_size,
            seed_state=frozen,
            stats=stats,
        )
        if selected is None:
            raise ValueError("unexpected fallback in admitted scale case")
        actual = compact_indices(selected.clamp_min(0), selected >= 0)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference[start:stop], atol=0, rtol=0)
        assert int(stats["seed_source"]) == 1
        if target_seed is None:
            target_seed = stats["seed_ids"][0].clone()
        assert (stats["seed_ids"] == target_seed[None]).all(), "seed set changed with query batch"
        count = stats["visible"] - stats["seed_count"]
        candidates += count.sum()
        survivors += stats["survivors"].sum()
        visible += stats["visible"].sum()
        pruned += (count - stats["survivors"]).sum()
        thresholds.append(stats["seed_threshold"].clone())
        survivor_counts.append(stats["survivors"].clone())
        del selected, actual, stats, frozen

    def baseline():
        result = None
        for start, stop in chunks:
            scores = dense_indexer_scores(query[start:stop], key, weights[start:stop])
            scores.masked_fill_(~causal[start:stop], torch.finfo(query.dtype).min)
            ids = dsa_topk_indices(scores)
            result = compact_indices(ids, causal[start:stop].gather(1, ids.long()))
        return result

    def optimized():
        result = None
        for start, stop in chunks:
            # The same preceding real query block supplies every batch and
            # replay. Current outputs never become witnesses within this audit.
            frozen = DoSASeedState(history.selected, history.visible_end, history.key_count)
            ids = dosa_select(
                query[start:stop],
                key,
                weights[start:stop],
                offsets[start:stop],
                heads=heads,
                seed_size=seed_size,
                seed_state=frozen,
            )
            result = compact_indices(ids.clamp_min(0), ids >= 0)
        return result

    samples = {"dsa": [], "dosa": []}
    for function in (baseline, optimized):
        for _ in range(2):
            function()
    torch.cuda.synchronize()
    for repeat in range(5):
        if time.monotonic() > deadline - 20:
            raise TimeoutError("shared GPU deadline during eager scale timing")
        functions = [("dsa", baseline), ("dosa", optimized)]
        for name, function in functions[:: 1 if repeat % 2 == 0 else -1]:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            function()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end))
    graph_samples = {}
    for name, function in [("dsa", baseline), ("dosa", optimized)]:
        if time.monotonic() > deadline - 30:
            raise TimeoutError("shared GPU deadline before graph scale timing")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = function()
        graph.replay()
        for value, reference in zip(output, expected):
            torch.testing.assert_close(value, reference[-len(value) :], atol=0, rtol=0)
        graph_samples[name] = []
        for _ in range(5):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            graph_samples[name].append(begin.elapsed_time(end))
        del graph, output
    dsa_ms = statistics.median(samples["dsa"])
    dosa_ms = statistics.median(samples["dosa"])
    dsa_graph_ms = statistics.median(graph_samples["dsa"])
    dosa_graph_ms = statistics.median(graph_samples["dosa"])
    nonseed = int(candidates)
    survival = int(survivors)
    visible_count = int(visible)
    # Logical-work estimates distinguish actual dense full-width scoring from
    # the paper's visible-pair model. Tensor Core row padding is separate.
    logical_work = total_queries * seed_size * 16 + nonseed * (heads + 1) + survival * 16
    return dict(
        N=width,
        total_queries=total_queries,
        batch_queries=batch_size,
        R=heads,
        S=seed_size,
        candidate_pairs=nonseed,
        visible_pairs=visible_count,
        pruned_pairs=int(pruned),
        prune_percent=100 * int(pruned) / nonseed,
        skipped_visible_percent=100 * int(pruned) / visible_count,
        seed_fraction_of_visible=total_queries * seed_size / visible_count,
        dsa_ms=dsa_ms,
        dosa_ms=dosa_ms,
        speedup=dsa_ms / dosa_ms,
        dsa_graph_ms=dsa_graph_ms,
        dosa_graph_ms=dosa_graph_ms,
        graph_speedup=dsa_graph_ms / dosa_graph_ms,
        logical_ratio_to_dense_width=logical_work / (total_queries * width * 16),
        logical_ratio_to_visible=logical_work / (visible_count * 16),
        screening_rows_logical=heads + 1,
        screening_rows_padded=max(4, 1 << heads.bit_length()),
        samples_ms=samples,
        graph_samples_ms=graph_samples,
        seed_ids=target_seed.cpu().tolist(),
        thresholds_sha256=hashlib.sha256(torch.cat(thresholds).cpu().numpy().tobytes()).hexdigest(),
        survivors_sha256=hashlib.sha256(torch.cat(survivor_counts).cpu().numpy().tobytes()).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=850)
    parser.add_argument("--head-audit", action="store_true")
    parser.add_argument("--lengths", nargs="+", type=int, default=[16384, 65536, 131072])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[128, 1024, 4096])
    parser.add_argument("--heads", nargs="+", type=int, default=[12])
    parser.add_argument("--audit-bounds", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.budget_seconds <= 900:
        parser.error("GPU budget must be within [1,900]")
    if any(not 16384 <= width <= 262144 for width in args.lengths):
        parser.error("lengths must be within [16384,262144]")
    if any(batch < 4 or batch > 4096 or 4096 % batch or batch % 4 for batch in args.batch_sizes):
        parser.error("batch sizes must divide4096 and be multiples of4")
    if any(not 1 <= heads < 16 for heads in args.heads):
        parser.error("head budgets must be within [1,15]")
    if args.head_audit and args.heads != [12]:
        parser.error("legacy --head-audit requires --heads12")
    deadline = START + args.budget_seconds
    args.output.mkdir(parents=True, exist_ok=True)
    entry = json.loads(args.manifest.read_text())["samples"][0]
    text = Path(entry["text_file"]).read_text()
    state = dict(cases=[], input=entry, scope="Keye layer0 scaling audit", gpu=torch.cuda.get_device_name())
    try:
        for width in args.lengths:
            if entry["tokens"] < width:
                raise ValueError("natural input shorter than requested prefix")
            if time.monotonic() > deadline - 60:
                break
            tensors = layer_zero(args.checkpoint, text, width, 5632)
            data = dict(
                iq=tensors["iq"][-4096:].clone().contiguous(),
                ik=tensors["ik"].clone(),
                iw=tensors["iw"][-4096:].clone().contiguous(),
                offsets=tensors["offsets"][-4096:].clone(),
            )
            previous_q = tensors["iq"][:-4096].clone().contiguous()
            previous_w = tensors["iw"][:-4096].clone().contiguous()
            previous_offsets = tensors["offsets"][:-4096].clone()
            del tensors
            history_ids = []
            for start in range(0, len(previous_q), 128):
                stop = min(start + 128, len(previous_q))
                causal = torch.arange(width, device="cuda")[None] <= previous_offsets[start:stop, None]
                scores = dense_indexer_scores(previous_q[start:stop], data["ik"], previous_w[start:stop])
                scores.masked_fill_(~causal, torch.finfo(scores.dtype).min)
                history_ids.append(dsa_topk_indices(scores).int())
            history = DoSASeedState()
            history.publish(torch.cat(history_ids), previous_offsets, width)
            del history_ids, previous_q, previous_w, previous_offsets, scores, causal
            data["causal"] = torch.arange(width, device="cuda")[None] <= data["offsets"][:, None]
            expected_ids = torch.empty((4096, 2048), device="cuda", dtype=torch.int32)
            expected_counts = torch.empty(4096, device="cuda", dtype=torch.int32)
            for start in range(0, 4096, 128):
                stop = start + 128
                scores = dense_indexer_scores(data["iq"][start:stop], data["ik"], data["iw"][start:stop])
                scores.masked_fill_(~data["causal"][start:stop], torch.finfo(scores.dtype).min)
                ids = dsa_topk_indices(scores)
                expected_ids[start:stop], expected_counts[start:stop] = compact_indices(
                    ids, data["causal"][start:stop].gather(1, ids.long())
                )
            del scores, ids
            expected = (expected_ids, expected_counts)
            same_seed = None
            same_thresholds = None
            survivor_hashes = {}
            certificate_audits = {}
            for batch_size in args.batch_sizes:
                for heads in args.heads:
                    if time.monotonic() > deadline - 60:
                        state["cases"].append(
                            dict(N=width, batch_queries=batch_size, R=heads, status="not_run_time_budget")
                        )
                        continue
                    if args.audit_bounds and heads not in certificate_audits:
                        certificate_audits[heads] = audit_certificate(data, history, heads, deadline)
                        logger.info(
                            "N%d R%d exhaustive bounds checked on%d candidate pairs",
                            width,
                            heads,
                            certificate_audits[heads]["checked_candidate_pairs"],
                        )
                        torch.cuda.empty_cache()
                    result = measure_batch(data, history, expected, batch_size, deadline, heads=heads)
                    if same_seed is None:
                        same_seed = result["seed_ids"]
                    assert result["seed_ids"] == same_seed, "seed differs across batching or head conditions"
                    if same_thresholds is None:
                        same_thresholds = result["thresholds_sha256"]
                    assert result["thresholds_sha256"] == same_thresholds, "seed thresholds changed"
                    if heads in survivor_hashes:
                        assert result["survivors_sha256"] == survivor_hashes[heads], (
                            "survivors changed with batch size"
                        )
                    survivor_hashes[heads] = result["survivors_sha256"]
                    if args.audit_bounds:
                        result["certificate_audit"] = certificate_audits[heads]
                    suffix = f"-r{heads}" if len(args.heads) > 1 or heads != 12 else ""
                    atomic_json(args.output / f"n{width}-q{batch_size}{suffix}.json", result)
                    state["cases"].append(dict(N=width, batch_queries=batch_size, R=heads, status="ok"))
                    state["elapsed_seconds"] = time.monotonic() - START
                    atomic_json(args.output / "status.json", state)
                    logger.info(
                        "N%d batchQ%d R%d pruning %.2f%% eager %.3f/%.3fms graph %.3f/%.3fms",
                        width,
                        batch_size,
                        heads,
                        result["prune_percent"],
                        result["dsa_ms"],
                        result["dosa_ms"],
                        result["dsa_graph_ms"],
                        result["dosa_graph_ms"],
                    )
                    torch.cuda.empty_cache()
            if args.head_audit and width == 131072:
                for heads in (4, 7, 8, 10):
                    if time.monotonic() > deadline - 60:
                        state["cases"].append(dict(N=width, batch_queries=4096, R=heads, status="not_run_time_budget"))
                        continue
                    result = measure_batch(data, history, expected, 4096, deadline, heads=heads)
                    assert result["seed_ids"] == same_seed
                    assert result["thresholds_sha256"] == same_thresholds
                    atomic_json(args.output / f"n{width}-q4096-r{heads}.json", result)
                    state["cases"].append(dict(N=width, batch_queries=4096, R=heads, status="ok"))
                    atomic_json(args.output / "status.json", state)
                    logger.info(
                        "Head audit R%d pruning %.2f%% eager %.3f/%.3fms graph %.3f/%.3fms",
                        heads,
                        result["prune_percent"],
                        result["dsa_ms"],
                        result["dosa_ms"],
                        result["dsa_graph_ms"],
                        result["dosa_graph_ms"],
                    )
                    torch.cuda.empty_cache()
            del data, expected, expected_ids, expected_counts, history
            torch.cuda.empty_cache()
    except Exception as error:
        state["error"] = f"{type(error).__name__}: {error}"
        logger.exception("Scale audit stopped")
        raise
    finally:
        state["elapsed_seconds"] = time.monotonic() - START
        atomic_json(args.output / "status.json", state)


if __name__ == "__main__":
    main()
