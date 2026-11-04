# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Capture real text-only Keye activations, loading one checkpoint layer at a
time.

Uses Relax's original DSA and sparse attention plus its TP1 MoE arithmetic.
This is an activation collector, NOT an end-to-end latency benchmark. No Ray
server or training task is started. Only the requested layers are serialized.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

from relax.models.keye_vl2.indexer import KeyeIndexer
from relax.models.keye_vl2.indexer_scores import dense_indexer_scores
from relax.models.keye_vl2.norm import _RMSNorm
from relax.models.keye_vl2.sglang.ops import expert_forward, rope
from relax.models.keye_vl2.sparse_attention import compact_indices, sparse_attention
from relax.models.keye_vl2.topk import dsa_topk_indices


@torch.no_grad()
def capture(args: argparse.Namespace) -> None:
    from sglang.srt import runtime_context
    from sglang.srt.server_args import ServerArgs

    root = args.checkpoint
    # Standalone kernel use needs configuration only, not a server/process group.
    runtime_context.publish(
        ServerArgs(model_path=str(root), trust_remote_code=True, context_length=args.length, tp_size=1),
        role="scheduler",
    )
    config = json.loads((root / "config.json").read_text())
    mapping = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    layers = sorted(set(args.layers))
    if args.length > config["max_position_embeddings"] or max(layers) >= config["num_hidden_layers"]:
        raise ValueError("Requested capture exceeds checkpoint configuration")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def weight(name: str, cuda: bool = True) -> torch.Tensor:
        with safe_open(root / mapping[name], framework="pt", device="cpu") as shard:
            value = shard.get_tensor(name)
        return value.cuda().bfloat16() if cuda else value

    tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True, local_files_only=True)
    tokens = tokenizer.encode(args.text.read_text())
    if not tokens:
        raise ValueError("Empty input text")
    ids = torch.tensor((tokens * ((args.length + len(tokens) - 1) // len(tokens)))[: args.length])
    hidden = weight("model.embed_tokens.weight", False)[ids].cuda().bfloat16()
    positions = torch.arange(args.length, device="cuda")
    eps, theta = config["rms_norm_eps"], config["rope_theta"]
    for layer in range(max(layers) + 1):
        prefix = f"model.layers.{layer}."
        normalized = _RMSNorm.apply(hidden, weight(prefix + "input_layernorm.weight"), eps)
        indexer = KeyeIndexer(config["hidden_size"], eps, theta).cuda().bfloat16().eval()
        indexer.load_state_dict(
            {name: weight(prefix + "self_attn.sa_indexer." + name) for name in indexer.state_dict()}
        )
        iq, ik, iw = indexer(normalized, positions)
        if layer in layers:
            destination = args.output_dir / f"layer-{layer}-n{args.length}.pt"
            torch.save(
                {
                    "iq": iq.cpu(),
                    "ik": ik.cpu(),
                    "iw": iw.cpu(),
                    "offsets": positions.cpu(),
                    "layer": layer,
                    "source": "checkpoint TP1 layer-streamed Relax DSA forward",
                },
                destination,
            )
            print(f"captured {destination}", flush=True)
        del indexer
        if layer == max(layers):
            break
        qkv = []
        for name, heads in (
            ("q", config["num_attention_heads"]),
            ("k", config["num_key_value_heads"]),
            ("v", config["num_key_value_heads"]),
        ):
            value = F.linear(normalized, weight(prefix + f"self_attn.{name}_proj.weight")).view(args.length, heads, -1)
            if name != "v":
                value = _RMSNorm.apply(value, weight(prefix + f"self_attn.{name}_norm.weight"), eps)
                value = rope(value, positions, theta)
            qkv.append(value)
        q, k, v = qkv
        attention = torch.empty_like(q)
        for start in range(0, args.length, args.attention_chunk):
            end = min(start + args.attention_chunk, args.length)
            causal = positions[None] <= positions[start:end, None]
            scores = dense_indexer_scores(iq[start:end], ik, iw[start:end]).masked_fill(
                ~causal, torch.finfo(iq.dtype).min
            )
            selected = dsa_topk_indices(scores)
            slots, counts = compact_indices(selected, causal.gather(1, selected.long()))
            attention[start:end] = sparse_attention(q[start:end], k, v, slots, counts)
        hidden = hidden + F.linear(attention.flatten(1), weight(prefix + "self_attn.o_proj.weight"))
        del qkv, q, k, v, attention, iq, ik, iw, normalized, value
        normalized = _RMSNorm.apply(hidden, weight(prefix + "post_attention_layernorm.weight"), eps)
        gate = weight(prefix + "mlp.gate.weight").double()
        gate_up = weight(prefix + "mlp.experts.gate_up_proj").transpose(1, 2).contiguous()
        down = weight(prefix + "mlp.experts.down_proj").transpose(1, 2).contiguous()
        for start in range(0, args.length, args.mlp_chunk):
            end = min(start + args.mlp_chunk, args.length)
            x = normalized[start:end]
            logits = F.linear(x.double(), gate).float()
            values, experts = torch.topk(logits, config["num_experts_per_tok"], dim=-1)
            probabilities = values.softmax(-1)
            hidden[start:end] += expert_forward(x, gate_up, down, probabilities, experts.int())
        del gate, gate_up, down, normalized, x
        torch.cuda.synchronize()
        print(f"completed layer {layer}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 23, 47])
    parser.add_argument("--attention-chunk", type=int, default=128)
    parser.add_argument("--mlp-chunk", type=int, default=512)
    capture(parser.parse_args())


if __name__ == "__main__":
    main()
