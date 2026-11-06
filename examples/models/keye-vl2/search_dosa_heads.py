# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only approximate head-swap search; GPU certificate validation is
separate.

This diagnostic replays layer-zero projections on CPU, considers at most two
rounds of 48 one-for-one swaps per query, and never asserts numerical safety
from CPU scores. The production selector must certify the chosen head sets.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from relax.utils.logging_utils import get_logger


os.environ["CUDA_VISIBLE_DEVICES"] = ""

logger = get_logger(__name__)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(8)
model = args.checkpoint
config = json.loads((model / "config.json").read_text())
mapping = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]


def weight(name: str) -> torch.Tensor:
    with safe_open(model / mapping[name], framework="pt", device="cpu") as shard:
        return shard.get_tensor(name)


def rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    inverse = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + config["rms_norm_eps"])
    return (x.float() * inverse * w.float()).bfloat16()


def rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    frequencies = 1 / config["rope_theta"] ** (torch.arange(0, 64, 2).float() / 64)
    angles = positions[:, None].float() * frequencies
    angles = torch.cat((angles, angles), -1)[:, None]
    first, second = x.chunk(2, -1)
    return x * angles.cos().bfloat16() + torch.cat((-second, first), -1) * angles.sin().bfloat16()


def hadamard(x: torch.Tensor) -> torch.Tensor:
    y, shape, step = (x.float(), x.shape, 1)
    while step < 64:
        blocks = y.reshape(*shape[:-1], -1, 2, step)
        first, second = (blocks[..., 0, :], blocks[..., 1, :])
        y = torch.stack((first + second, first - second), -2).reshape(shape)
        step *= 2
    return (y * 0.125).bfloat16()


def hidden(ids: torch.Tensor) -> torch.Tensor:
    return rms(emb[ids], inorm)


def projectq(ids: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    projected = F.linear(hidden(ids), params["q_proj.weight"])
    query, weights = projected.split([1024, 16], -1)
    query = query.reshape(-1, 16, 64)
    inverse = torch.rsqrt(query.float().square().mean(-1, keepdim=True) + config["rms_norm_eps"])
    query = params["q_norm.weight"] * (query.float() * inverse).bfloat16()
    return (hadamard(rope(query, positions)), weights)


def projectk(ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    key = F.linear(hidden(ids), params["k_proj.weight"])[:, None]
    key = F.layer_norm(
        key.float(), (64,), params["k_norm.weight"].float(), params["k_norm.bias"].float(), config["rms_norm_eps"]
    ).bfloat16()
    return hadamard(rope(key, positions))[:, 0]


emb = weight("model.embed_tokens.weight")
inorm = weight("model.layers.0.input_layernorm.weight")
prefix = "model.layers.0.self_attn.sa_indexer."
parameter_names = ["q_proj.weight", "k_proj.weight", "q_norm.weight", "k_norm.weight", "k_norm.bias"]
params = {key: weight(prefix + key) for key in parameter_names}


@torch.no_grad()
def evaluate(
    x: torch.Tensor,
    w: torch.Tensor,
    keys: torch.Tensor,
    frequency: torch.Tensor,
    combinations: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit each candidate and evaluate the complete approximate score bound."""
    ids = torch.tensor(combinations, dtype=torch.long)
    vectors = x[ids]
    gh = torch.einsum("hd,cjd->chj", x, vectors)
    gram = vectors @ vectors.transpose(1, 2)
    diagonal = gram.diagonal(dim1=1, dim2=2).clamp_min(2 ** (-64))
    quality = gh.clamp_min(0) / diagonal.sqrt()[:, None]
    assignment = quality.argmax(-1)
    beta = torch.zeros_like(gh)
    initial = (gh.clamp_min(0) / diagonal[:, None]).clamp_max(65536)
    beta.scatter_(2, assignment[:, :, None], initial.gather(2, assignment[:, :, None]))
    retained_mask = torch.zeros((len(ids), 16), dtype=torch.bool).scatter_(1, ids, True)
    omitted = (w > 0)[None] & ~retained_mask
    beta *= omitted[:, :, None]
    # Match the bounded nonnegative fitter; this is not a numerical certificate.
    for _ in range(12):
        for j in range(4):
            derivative = gh[:, :, j] - (beta * gram[:, None, :, j]).sum(-1)
            beta[:, :, j] = (beta[:, :, j] + derivative / diagonal[:, None, j]).clamp(0, 65536) * omitted
    residual = x[None] - beta @ vectors
    weighted = residual * (w[None] * omitted)[:, :, None]
    envelope = weighted.abs().sum(1)
    arithmetic = (w.abs()[:, None] * x.abs()).sum(0) / 64
    joint = 0.5 * (weighted.sum(1).norm(dim=-1) + (weighted @ weighted.transpose(1, 2)).abs().sum((1, 2)).sqrt())
    allowance = torch.minimum((envelope + arithmetic).norm(dim=-1), joint + arithmetic.norm())
    support = (w < 0)[None] & ~retained_mask & (frequency >= 0.5)[None]
    aggregate = ((-w[None] * support)[:, :, None] * x[None]).sum(1)
    aggregate_bf16 = aggregate.bfloat16().float()
    folded = w[ids] + (beta * (w[None] * omitted)[:, :, None]).sum(1)
    dot_error = aggregate_bf16.norm(dim=-1) + (folded.abs() * vectors.norm(dim=-1)).sum(1)
    allowance += (aggregate - aggregate_bf16).norm(dim=-1) + dot_error * 2 ** (-15)
    retained = ((vectors @ keys.T).relu() * folded[:, :, None]).sum(1)
    negative = (aggregate_bf16 @ keys.T).relu()
    bounds = (retained - negative + allowance[:, None] * keys.norm(dim=-1)[None]) / 8
    return (bounds, weighted.norm(dim=-1).sum(1))


args.output.mkdir(parents=True, exist_ok=True)
manifest = json.loads((args.baseline / "input-manifest.json").read_text())
all_records = []
with torch.no_grad():
    for n in (8192, 16384):
        for entry in manifest["samples"]:
            started = time.monotonic()
            name = f"{entry['label']}-n{n}"
            baseline = json.loads((args.baseline / f"{name}.json").read_text())
            if entry["repeated"]:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
                tokens = tokenizer.encode(Path(entry["text_file"]).read_text())
                tokens = (tokens * ((n + len(tokens) - 1) // len(tokens)))[:n]
            else:
                tokens = np.load(entry["token_file"])[:n].tolist()
            tokens = torch.tensor(tokens)
            positions = torch.arange(n - 128, n)
            q, weights = projectq(tokens[positions], positions)
            audit = np.load(args.baseline / f"{name}-mapping.npz")
            seeds = torch.tensor(audit["seeds"][:, :128]).long()
            seed_keys = projectk(tokens[seeds.reshape(-1)], seeds.reshape(-1)).reshape(128, 128, 64)
            ranks = np.empty((128, 16), dtype=np.int32)
            records = []
            for row in range(128):
                x, w = (q[row].float(), weights[row].float())
                before_ids = torch.tensor(audit["projections"][row, :, 19]).argsort()[:4].tolist()
                current = before_ids[:]
                visible = n - 127 + row
                sampled_ids = torch.linspace(0, visible - 1, 128).long()
                keys = projectk(tokens[sampled_ids], sampled_ids).float()
                frequency = (x @ seed_keys[row].float().T > 0).float().mean(-1)
                tau = baseline["rows"][row]["tau_seed"]
                initial_bound, initial_residual = evaluate(x, w, keys, frequency, [current])
                objectives = [float(initial_bound.mean())]
                changed = 0
                for _ in range(2):
                    candidates = [current]
                    for slot in range(4):
                        for head in range(16):
                            if head not in current:
                                trial = current[:]
                                trial[slot] = head
                                candidates.append(trial)
                    assert len(candidates) == 49
                    assert len({tuple(sorted(c)) for c in candidates}) == 49
                    bounds, residual_norms = evaluate(x, w, keys, frequency, candidates)
                    if not bool(torch.isfinite(bounds).all()):
                        raise ValueError("nonfinite approximate CPU bound; search stopped")
                    # Ties retain the incumbent; the exact column order is replayed on GPU.
                    best = min(
                        range(len(candidates)),
                        key=lambda j: (-int((bounds[j] < tau).sum()), float(bounds[j].mean()), j),
                    )
                    if best == 0:
                        break
                    current = candidates[best]
                    changed += 1
                    objectives.append(float(bounds[best].mean()))
                final_bound, final_residual = evaluate(x, w, keys, frequency, [current])
                order = current + [h for h in range(16) if h not in current]
                for rank, head in enumerate(order):
                    ranks[row, head] = rank
                records.append(
                    dict(
                        query=row,
                        initial_heads=before_ids,
                        chosen_heads=current,
                        accepted_swaps=changed,
                        mean_bound_history=objectives,
                        cpu_initial_mean_bound=float(initial_bound.mean()),
                        cpu_final_mean_bound=float(final_bound.mean()),
                        cpu_initial_residual_sum=float(initial_residual[0]),
                        cpu_final_residual_sum=float(final_residual[0]),
                        chosen_negative_heads=sum((float(w[h]) < 0 for h in current)),
                        cpu_sample_pruned=int((final_bound < tau).sum()),
                    ),
                )
            np.save(args.output / f"{name}-ranks.npy", ranks)
            result = dict(
                case=name,
                rounds=2,
                candidates_per_round=49,
                sampled_keys=128,
                objective="maximize sampled certified-style rejection, then minimize mean approximate U",
                cpu_seconds=time.monotonic() - started,
                rows=records,
            )
            (args.output / f"{name}-search.json").write_text(json.dumps(result, indent=2))
            all_records.append(result)
            logger.info(
                "Search %s: %.2fs, changed %d/128 queries",
                name,
                result["cpu_seconds"],
                sum((r["initial_heads"] != r["chosen_heads"] for r in records)),
            )
(args.output / "search-summary.json").write_text(
    json.dumps(
        dict(
            cases=len(all_records),
            cpu_case_seconds=sum((r["cpu_seconds"] for r in all_records)),
            note="Approximate CPU replay; no global-optimum/speedup claim; GPU recertification required.",
        ),
        indent=2,
    ),
)
