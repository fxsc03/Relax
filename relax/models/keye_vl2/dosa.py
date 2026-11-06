# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in exact Keye selection; callers retain ownership of DSA fallback.

The result contains original document-local token IDs, with -1 for unused
entries. No model weights, attention gradients or cache ownership change.
"""

from typing import Any

import torch

from .dosa_seed import DoSASeedState


def backend_config(source: Any) -> tuple[str, int]:
    backend = getattr(source, "keye_dsa_backend", "dsa")
    heads = getattr(source, "keye_dosa_heads", 4)
    if backend not in ("dsa", "dosa"):
        raise ValueError("keye_dsa_backend must be dsa or dosa")
    if not isinstance(heads, int) or not 1 <= heads < 16:
        raise ValueError("keye_dosa_heads must be in [1, 15]")
    return backend, heads


def frozen_indexer_enabled(config: Any, indexer: torch.nn.Module) -> bool:
    # In particular, no-grad during reentrant checkpoint's first forward
    # must not send a *trainable* indexer through a different selector.
    return getattr(config, "keye_dsa_backend", "dsa") == "dosa" and not any(
        p.requires_grad for p in indexer.parameters()
    )


@torch.no_grad()
def dosa_select(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    valid_keys: torch.Tensor | None = None,
    *,
    heads: int = 4,
    seed_size: int | None = None,
    seed_policy: str = "informative",
    seed_state: DoSASeedState | None = None,
    stats: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor | None:
    """Return exact IDs, or None to request the caller's original DSA path.

    ``offsets`` are causal token offsets, never RoPE/mRoPE coordinates.
    ``valid_keys`` is document-local. CUDA summaries live for this call only.
    Diagnostics remain on the GPU; only a benchmark should copy them to CPU.
    """
    if (
        query.device.type != "cuda"
        or torch.version.hip is not None
        or query.dtype != torch.bfloat16
        or key.dtype != query.dtype
        or weights.dtype != query.dtype
        or query.shape[1:] != (16, 64)
        or key.shape[1:] != (1, 64)
        or weights.shape != query.shape[:2]
        or key.device != query.device
        or weights.device != query.device
        or not isinstance(heads, int)
        or not 1 <= heads < 16
        or torch.cuda.get_device_capability(query.device)[0] not in (8, 9)
    ):
        return None
    if seed_policy not in ("uniform", "informative"):
        raise ValueError("seed_policy must be uniform or informative")
    if seed_size is None:
        seed_size = 12288 if seed_policy == "informative" else 4096
    rows, width = query.shape[0], key.shape[0]
    # Bound integer seed arithmetic, and exclude zero-width attention from
    # this optimization (the caller owns its existing empty-input behavior).
    if (
        seed_size not in (2048, 4096, 8192, 12288)
        or width == 0
        or width > 262144
        or offsets.shape != (rows,)
        or offsets.device != query.device
        or offsets.dtype not in (torch.int32, torch.int64)
    ):
        return None
    if valid_keys is not None and (
        valid_keys.shape != (width,) or valid_keys.dtype != torch.bool or valid_keys.device != query.device
    ):
        return None
    tokens = torch.arange(width, device=query.device)
    valid = torch.ones_like(tokens, dtype=torch.bool) if valid_keys is None else valid_keys
    if width <= 2048:
        visible = (tokens[None] <= offsets[:, None]) & valid[None]
        if stats is not None:
            counts = visible.sum(-1, dtype=torch.int32)
            zero = torch.zeros_like(counts)
            stats.update(
                visible=counts,
                seed_count=zero,
                survivors=zero,
                unbounded=zero,
                exact_score_rejected=zero,
                selection_candidates=counts,
            )
        result = tokens.expand(rows, -1).masked_fill(~visible, -1)
        if rows and seed_state is not None and seed_policy == "informative":
            seed_state.publish(result, offsets, width)
        return result
    # All keys would be seeds; no screening is possible. H20 measurements
    # also show the original selector is faster here (see the validation note).
    if width <= seed_size:
        return None

    from . import dosa_kernels as kernels
    from . import dosa_parallel as parallel

    query, key, weights = query.contiguous(), key.contiguous(), weights.contiguous()
    valid_ids = tokens.masked_fill(~valid, width).sort().values.int()
    visible = torch.searchsorted(valid_ids, offsets.clamp(max=width - 1).int(), right=True).int()
    seeds = torch.empty((rows, seed_size), device=query.device, dtype=torch.int32)
    bases = torch.empty(((rows + 3) // 4,), device=query.device, dtype=torch.int32)
    seed_scores = query.new_empty((rows, seed_size))
    plan = torch.empty((rows, heads * 66 + 193), device=query.device, dtype=torch.float32)
    result = torch.empty((rows, 2048), device=query.device, dtype=torch.int32)
    capacity = width - seed_size  # Capacity covers ALL non-seed keys.
    tiles = (capacity + 127) // 128
    vectors = query.new_zeros((rows, heads + 1, 64))
    coefficients = torch.zeros((rows, heads), device=query.device, dtype=torch.float32)
    error = torch.full((rows,), float("inf"), device=query.device)
    norms = torch.empty((width,), device=query.device)
    threshold = torch.empty((rows,), device=query.device)
    flags = torch.empty((rows, capacity), device=query.device, dtype=torch.bool)
    counts = torch.empty((rows, tiles), device=query.device, dtype=torch.int32)
    unsafe = torch.empty_like(counts)
    candidate_ids = torch.empty((rows, capacity), device=query.device, dtype=torch.int32)
    words = torch.empty((rows, capacity + 2048), device=query.device, dtype=torch.uint32)
    low = torch.empty((rows, capacity + 2048), device=query.device, dtype=torch.uint8)
    lengths = torch.empty((rows,), device=query.device, dtype=torch.int32)
    boundary_ids = torch.empty((rows, capacity + 2048), device=query.device, dtype=torch.uint32)
    launch = {"num_warps": 4, "enable_fp_fusion": False}
    grouped = seed_policy == "informative"
    ordered_ids = valid_ids
    if rows and grouped:
        from .dosa_seed import informative_order

        ordered_ids, seed_source, votes = informative_order(
            query, key, weights, offsets, valid, valid_ids, visible, seed_size, seed_state
        )
    if rows:
        parallel.make_seeds[(rows, seed_size // 128)](
            ordered_ids, visible, seeds, bases, width, rows, seed_size, ORIGINAL=valid_ids, GROUPED=grouped, **launch
        )
        kernels.seed_scores[(rows, seed_size // 128)](
            query, key, weights, seeds, visible, seed_scores, query.stride(0), weights.stride(0), seed_size, **launch
        )
        from .topk import dsa_topk_indices

        seed_top = dsa_topk_indices(seed_scores).contiguous()
        kernels.prepare[(rows,)](
            query, weights, key, seeds, visible, plan, query.stride(0), weights.stride(0), seed_size, heads, **launch
        )
        import triton

        parallel.seed_codes[(rows,)](
            seed_scores,
            seeds,
            seed_top,
            visible,
            valid_ids,
            None,
            threshold,
            seed_size,
            capacity,
            WORDS=words,
            LOW=low,
            LENGTHS=lengths,
            PACKED=True,
            **launch,
        )
        parallel.pack_plan[(rows,)](plan, visible, vectors, coefficients, error, heads, seed_size, **launch)
        parallel.key_norm[((width + 127) // 128,)](key, norms, width, **launch)
        parallel.screen[((rows + 3) // 4, tiles)](
            key,
            norms,
            vectors,
            coefficients,
            error,
            threshold,
            bases,
            ordered_ids,
            visible,
            flags,
            counts,
            unsafe,
            rows,
            width,
            seed_size,
            heads,
            capacity,
            tiles,
            4 * max(4, triton.next_power_of_2(heads + 1)),
            GROUPED=grouped,
            **launch,
        )
        prefix = torch.cumsum(counts, -1, dtype=torch.int32)
        parallel.compact[(rows, tiles)](
            flags, prefix, bases, ordered_ids, candidate_ids, capacity, tiles, seed_size, GROUPED=grouped, **launch
        )
        parallel.refine[(rows, tiles)](
            query, key, weights, candidate_ids, prefix, threshold, words, low, lengths, capacity, tiles, **launch
        )
        parallel.select_records[(rows,)](words, low, lengths, boundary_ids, result, capacity + 2048, 512, **launch)
    if rows and seed_state is not None and grouped:
        seed_state.publish(result, offsets, width)
    if stats is not None:
        stats.update(
            seed_ids=seeds,
            seed_order=ordered_ids,
            seed_bases=bases,
            seed_threshold=threshold,
            visible=visible,
            seed_count=torch.where(visible <= 2048, 0, visible.clamp(max=seed_size)),
            survivors=prefix[:, -1] if rows else visible,
            unbounded=unsafe.sum(-1),
            refinement_tiles=(prefix[:, -1] + 127) // 128 if rows else visible,
            exact_score_rejected=prefix[:, -1] + visible.clamp(max=2048) - lengths if rows else visible,
            selection_candidates=lengths,
            seed_tiles=torch.where(visible > 2048, (visible.clamp(max=seed_size) + 127) // 128, 0),
        )
        if grouped and rows:
            stats.update(seed_source=seed_source, seed_votes=votes)
    return result
