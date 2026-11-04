# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Small selector benchmark; never starts a server, Ray job or training run.

Accepts real post-RoPE/Hadamard iq/ik/iw captures via --inputs. Alternatively
--checkpoint builds authentic *layer 0* inputs from token embeddings and the
input norm. It does not mislabel these as deeper-layer activation captures.
"""

import argparse
import importlib
import json
import statistics
import sys
import types
from contextlib import ExitStack
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import torch

from relax.models.keye_vl2.dosa import dosa_select
from relax.models.keye_vl2.indexer_scores import dense_indexer_scores
from relax.models.keye_vl2.sparse_attention import compact_indices
from relax.models.keye_vl2.topk import dsa_topk_indices


def time_ms(function: Callable, repeats: int) -> float:
    function()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


@torch.no_grad()
def layer_zero(checkpoint: Path, text: str, length: int, rows: int) -> dict[str, torch.Tensor]:
    from safetensors import safe_open
    from transformers import AutoTokenizer

    from relax.models.keye_vl2.indexer import KeyeIndexer
    from relax.models.keye_vl2.norm import _RMSNorm

    config = json.loads((checkpoint / "config.json").read_text())
    mapping = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(name: str) -> torch.Tensor:
        with safe_open(checkpoint / mapping[name], framework="pt", device="cpu") as shard:
            return shard.get_tensor(name)

    tokens = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True).encode(text)
    if not tokens:
        raise ValueError("Text must tokenize to at least one token")
    ids = torch.tensor((tokens * ((length + len(tokens) - 1) // len(tokens)))[:length])
    hidden = weight("model.embed_tokens.weight")[ids].cuda().bfloat16()
    norm = weight("model.layers.0.input_layernorm.weight").cuda().bfloat16()
    hidden = _RMSNorm.apply(hidden, norm, config["rms_norm_eps"])
    indexer = KeyeIndexer(config["hidden_size"], config["rms_norm_eps"], config["rope_theta"])
    indexer.load_state_dict(
        {name: weight("model.layers.0.self_attn.sa_indexer." + name) for name in indexer.state_dict()}
    )
    indexer = indexer.cuda().bfloat16().eval()
    positions = torch.arange(length, device="cuda")
    iq, ik, iw = indexer(hidden, positions)
    return {"iq": iq[-rows:], "ik": ik, "iw": iw[-rows:], "offsets": positions[-rows:]}


@torch.no_grad()
def measure(
    data: dict[str, torch.Tensor], heads: int, repeats: int, selector=dosa_select, kernel_modules=None, seed_size=4096
) -> dict:
    q, k, w = (data[name].cuda().contiguous() for name in ("iq", "ik", "iw"))
    offsets = data["offsets"].cuda()
    valid = data.get("valid")
    valid = torch.ones(len(k), device="cuda", dtype=torch.bool) if valid is None else valid.cuda()
    causal = (torch.arange(len(k), device="cuda")[None] <= offsets[:, None]) & valid[None]

    def baseline():
        scores = dense_indexer_scores(q, k, w).masked_fill(~causal, torch.finfo(q.dtype).min)
        selected = dsa_topk_indices(scores)
        return compact_indices(selected, causal.gather(1, selected.long()))

    fallback = False
    saved_baseline = kernel_modules is not None

    def optimized(stats=None):
        nonlocal fallback
        options = {} if saved_baseline else {"seed_size": seed_size}
        selected = selector(q, k, w, offsets, valid, heads=heads, stats=stats, **options)
        if selected is None:
            fallback = True
            if stats is not None:
                counts = causal.sum(-1, dtype=torch.int32)
                zero = torch.zeros_like(counts)
                stats.update(visible=counts, seed_count=zero, survivors=zero, unbounded=zero)
            return baseline()
        return compact_indices(selected.clamp_min(0), selected >= 0)

    stats = {}
    actual, expected = optimized(stats), baseline()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    # Interleave measurements to reduce bias from changing colocated load.
    for _ in range(3):
        baseline()
        optimized()
    timings = {"dsa": [], "dosa": []}
    for iteration in range(repeats):
        functions = [("dsa", baseline), ("dosa", optimized)]
        for name, function in functions[:: 1 if iteration % 2 == 0 else -1]:
            timings[name].append(time_ms(function, 1))
    baseline_ms = statistics.median(timings["dsa"])
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    time_ms(optimized, 1)
    dosa_ms = statistics.median(timings["dosa"])
    peak = torch.cuda.max_memory_allocated() - allocated
    # Diagnostic wrappers measure launch intervals, not an additive breakdown.
    # The old scan remains fused; the new path exposes individual stages.
    from relax.models.keye_vl2 import dosa_kernels, dosa_parallel

    kernel_modules = kernel_modules or [dosa_kernels, dosa_parallel]
    metadata = {}

    events = {}

    class TimedKernel:
        def __init__(self, name, kernel):
            self.name, self.kernel = name, kernel

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                result = self.kernel[grid](*args, **kwargs)
                end.record()
                events[self.name] = (start, end)
                metadata[self.name] = {
                    "registers": result.n_regs,
                    "spill_slots": result.n_spills,
                    "shared_bytes": result.metadata.shared,
                }
                return result

            return launch

    with ExitStack() as stack:
        for module in kernel_modules:
            for name in (
                "seed_scores",
                "seed_topk",
                "prepare",
                "select_stream",
                "seed_codes",
                "pack_plan",
                "key_norm",
                "screen",
                "compact",
                "refine",
                "decode",
                "make_seeds",
                "radix_select",
                "selection_keys",
            ):
                if hasattr(module, name):
                    stack.enter_context(patch.object(module, name, TimedKernel(name, getattr(module, name))))

        def timed_function(name, function):
            def call(*args, **kwargs):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                result = function(*args, **kwargs)
                end.record()
                event_name = name
                suffix = 1
                while event_name in events:
                    event_name = f"{name}_{suffix}"
                    suffix += 1
                events[event_name] = (start, end)
                return result

            return call

        from relax.models.keye_vl2 import topk

        stack.enter_context(patch.object(topk, "dsa_topk_indices", timed_function("seed_topk", topk.dsa_topk_indices)))
        stack.enter_context(patch.object(torch, "topk", timed_function("integer_topk", torch.topk)))
        stack.enter_context(
            patch.object(dosa_parallel, "select_codes", timed_function("final_topk", dosa_parallel.select_codes))
        )
        stack.enter_context(patch.object(torch, "cumsum", timed_function("prefix_sum", torch.cumsum)))
        optimized()
        torch.cuda.synchronize()
    stages = {name: start.elapsed_time(end) for name, (start, end) in events.items()}
    counts = {name: value.cpu().tolist() for name, value in stats.items()}
    remaining = sum(max(count - seed_size, 0) for count in counts["visible"])
    return dict(
        gpu=torch.cuda.get_device_name(),
        timing_samples_ms=timings,
        timing_range_ms={name: [min(values), max(values)] for name, values in timings.items()},
        dtype=str(q.dtype),
        queries=len(q),
        keys=len(k),
        heads=heads,
        seed_capacity=seed_size,
        dsa_ms=baseline_ms,
        dosa_ms=dosa_ms,
        speedup=baseline_ms / dosa_ms,
        peak_extra_bytes=peak,
        nonseed_pruning_fraction=1 - sum(counts["survivors"]) / remaining if remaining and not fallback else None,
        original_backend_fallback=fallback,
        certificate_bypass_row_fraction=sum(n > 0 for n in counts["unbounded"]) / max(len(q), 1),
        counters=counts,
        kernel_launch_ms=stages,
        kernel_resources=metadata,
        visible_pruning_fraction=(remaining - sum(counts["survivors"])) / max(sum(counts["visible"]), 1),
        seed_fraction=sum(counts["seed_count"]) / max(sum(counts["visible"]), 1),
        refinement_tile_fill=(
            sum(counts["survivors"]) / max(128 * sum(counts.get("refinement_tiles", [])), 1)
            if "refinement_tiles" in counts
            else None
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", type=Path, help="torch.save dictionary or list of layer dictionaries")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline-dir", type=Path, help="Saved pre-change directory containing dosa.py and kernels")
    parser.add_argument("--text", type=Path, help="Prompt/tool history text for layer-zero input")
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed-size", type=int, default=4096, choices=[2048, 4096, 8192])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.baseline_dir and args.seed_size != 4096:
        parser.error("The saved v1 baseline has a fixed seed capacity of 4096")
    if args.inputs:
        data = torch.load(args.inputs, map_location="cpu", weights_only=True)
        source = str(args.inputs)
    elif args.checkpoint:
        text = (
            args.text.read_text()
            if args.text
            else "You are in a kitchen. Your task is to put a clean mug in the sink."
        )
        data = layer_zero(args.checkpoint, text, args.length, args.rows)
        source = "checkpoint layer 0; repeated input text (not deeper-layer activations)"
    else:
        torch.manual_seed(293)
        data = {
            "iq": torch.randn(args.rows, 16, 64, device="cuda", dtype=torch.bfloat16),
            "ik": torch.randn(args.length, 1, 64, device="cuda", dtype=torch.bfloat16),
            "iw": torch.randn(args.rows, 16, device="cuda", dtype=torch.bfloat16),
            "offsets": torch.arange(args.length - args.rows, args.length, device="cuda"),
        }
        source = "random tensors"
    selector, modules = dosa_select, None
    if args.baseline_dir:
        package = types.ModuleType("keye_dosa_saved")
        package.__path__ = [str(args.baseline_dir.resolve())]
        sys.modules[package.__name__] = package
        selector = importlib.import_module("keye_dosa_saved.dosa").dosa_select
        modules = [importlib.import_module("keye_dosa_saved.dosa_kernels")]
    reports = []
    for layer in data if isinstance(data, list) else [data]:
        if args.inputs:
            # Full-sequence captures can be replayed at multiple causal lengths.
            layer = dict(layer)
            n = min(args.length, len(layer["ik"]))
            if len(layer["iq"]) == len(layer["ik"]):
                for name in ("iq", "iw", "offsets"):
                    layer[name] = layer[name][max(n - args.rows, 0) : n]
            layer["ik"] = layer["ik"][:n]
            if "valid" in layer:
                layer["valid"] = layer["valid"][:n]
        report = measure(layer, args.heads, args.repeats, selector, modules, args.seed_size)
        report["layer"] = layer.get("layer", 0 if args.checkpoint else None)
        reports.append(report)
    args.output.write_text(json.dumps({"source": source, "results": reports}, indent=2) + "\n")
    for report in reports:
        print(f"DSA={report['dsa_ms']:.3f} ms DoSA={report['dosa_ms']:.3f} ms speedup={report['speedup']:.3f}x")


if __name__ == "__main__":
    main()
