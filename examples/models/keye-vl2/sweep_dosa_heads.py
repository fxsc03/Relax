# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Actual-path head-budget sweep with original BF16 scoring and exact
selection.

The production planner uses the paper best-single-anchor projection. The
default matrix is R=8/10/12 at 8K/16K, using the original input manifest.
"""

import argparse
import importlib.util
import json
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from benchmark_dosa import layer_zero
from diagnose_dosa import atomic_json, diagnose, summarize

from relax.models.keye_vl2.dosa import dosa_select
from relax.models.keye_vl2.indexer_scores import dense_indexer_scores
from relax.models.keye_vl2.sparse_attention import compact_indices
from relax.models.keye_vl2.topk import dsa_topk_indices
from relax.utils.logging_utils import get_logger


START = time.monotonic()
logger = get_logger(__name__)
BEFORE_SELECTOR = None


def load_before(directory: Path):
    """Load a recorded pre-change selector in its own relative-import
    package."""
    name = "relax.models.keye_vl2._dosa_before"
    package = types.ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    for module in ("indexer_scores", "topk"):
        sys.modules[f"{name}.{module}"] = __import__(f"relax.models.keye_vl2.{module}", fromlist=[module])
    for module in ("dosa_kernels", "dosa_parallel", "dosa"):
        spec = importlib.util.spec_from_file_location(f"{name}.{module}", directory / f"before-{module}.py")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = loaded
        spec.loader.exec_module(loaded)
        setattr(package, module, loaded)
    return loaded.dosa_select


@torch.no_grad()
def timing(data: dict, heads: int, deadline: float) -> dict:
    q, k, w = (data[name].contiguous() for name in ("iq", "ik", "iw"))
    offsets = data["offsets"]
    causal = torch.arange(len(k), device=q.device)[None] <= offsets[:, None]

    def baseline():
        scores = dense_indexer_scores(q, k, w).masked_fill(~causal, torch.finfo(q.dtype).min)
        selected = dsa_topk_indices(scores)
        return compact_indices(selected, causal.gather(1, selected.long()))

    def optimized():
        selected = dosa_select(q, k, w, offsets, heads=heads, seed_size=4096, seed_policy="uniform")
        if selected is None:
            raise ValueError("unexpected selector fallback")
        return compact_indices(selected.clamp_min(0), selected >= 0)

    def before():
        selected = BEFORE_SELECTOR(q, k, w, offsets, heads=heads, seed_size=4096)
        return compact_indices(selected.clamp_min(0), selected >= 0)

    actual, expected = optimized(), baseline()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    if BEFORE_SELECTOR is not None:
        for a, b in zip(before(), expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    for _ in range(3):
        baseline()
        optimized()
        if BEFORE_SELECTOR is not None:
            before()
    torch.cuda.synchronize()
    samples = {"dsa": [], "dosa": []}
    if BEFORE_SELECTOR is not None:
        samples["before"] = []
    for repeat in range(5):
        if time.monotonic() > deadline - 10:
            raise TimeoutError("shared GPU deadline during timing")
        functions = [("dsa", baseline), ("dosa", optimized)]
        if BEFORE_SELECTOR is not None:
            functions.append(("before", before))
        for name, function in functions[:: 1 if repeat % 2 == 0 else -1]:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
    dsa, dosa = (statistics.median(samples[name]) for name in ("dsa", "dosa"))
    result = dict(samples_ms=samples, dsa_ms=dsa, dosa_ms=dosa, speedup=dsa / dosa)
    if BEFORE_SELECTOR is not None:
        result["before_ms"] = statistics.median(samples["before"])
        result["before_to_after_speedup"] = result["before_ms"] / dosa
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before-dir", type=Path, help="Directory containing pre-change source snapshots")
    parser.add_argument("--budget-seconds", type=int, default=870)
    parser.add_argument("--heads", nargs="+", type=int, choices=range(1, 16), default=[8, 10, 12])
    args = parser.parse_args()
    if not 1 <= args.budget_seconds <= 900:
        parser.error("GPU budget must be in [1,900]")
    global BEFORE_SELECTOR
    if args.before_dir is not None:
        BEFORE_SELECTOR = load_before(args.before_dir)
    deadline = START + args.budget_seconds
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    state = dict(cases=[], gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())
    try:
        from fast_hadamard_transform import hadamard_transform

        check = hadamard_transform(torch.ones((4, 64), device="cuda", dtype=torch.bfloat16), scale=0.125)
        torch.cuda.synchronize()
        assert float(check[0, 0]) == 8
        for n in (8192, 16384):
            for entry in manifest["samples"]:
                if entry["tokens"] < n:
                    raise ValueError("input shorter than requested natural prefix")
                data = layer_zero(args.checkpoint, Path(entry["text_file"]).read_text(), n, 128)
                for heads in args.heads:
                    name = f"{entry['label']}-n{n}-r{heads}"
                    item = dict(case=name, heads=heads, length=n, status="not_run_time_budget")
                    state["cases"].append(item)
                    if time.monotonic() > deadline - 30:
                        continue
                    rows, audit = diagnose(data, deadline, heads=heads)
                    measured = timing(data, heads, deadline)
                    result = dict(
                        input=entry,
                        layer=0,
                        length=n,
                        heads=heads,
                        rows=rows,
                        summary=summarize(rows),
                        timing=measured,
                        projection_policy="best_single_anchor",
                    )
                    np.savez_compressed(args.output / f"{name}-mapping.npz", **audit)
                    atomic_json(args.output / f"{name}.json", result)
                    item["status"] = "ok"
                    state["elapsed_seconds"] = time.monotonic() - START
                    atomic_json(args.output / "status.json", state)
                    logger.info(
                        "%s pruning %.3f%%, DSA %.3fms, DoSA %.3fms, speedup %.3fx",
                        name,
                        result["summary"]["pruned_fraction"] * 100,
                        measured["dsa_ms"],
                        measured["dosa_ms"],
                        measured["speedup"],
                    )
                del data
                torch.cuda.empty_cache()
    except Exception as error:
        state["error"] = f"{type(error).__name__}: {error}"
        if state["cases"]:
            state["cases"][-1]["status"] = "error"
        logger.exception("Sweep stopped without automatic retry")
        raise
    finally:
        existing = {item["case"] for item in state["cases"]}
        for n in (8192, 16384):
            for entry in manifest["samples"]:
                for heads in args.heads:
                    name = f"{entry['label']}-n{n}-r{heads}"
                    if name not in existing:
                        state["cases"].append(dict(case=name, heads=heads, length=n, status="not_run_after_error"))
        state["elapsed_seconds"] = time.monotonic() - START
        atomic_json(args.output / "status.json", state)


if __name__ == "__main__":
    main()
