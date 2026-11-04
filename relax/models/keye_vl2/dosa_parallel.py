# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Parallel, capacity-safe DoSA screening and cross-tile refinement.

The screening dot products are BF16 tensor-core operations. Their FP32
accumulation error and the aggregate's BF16 conversion error are included in a
directed-rounding L2 certificate; refinement still uses score_tile.
"""

from typing import TYPE_CHECKING

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

from .dosa_kernels import _code, _safe, _up
from .indexer_scores import score_tile


if TYPE_CHECKING:
    import torch


@triton.jit
def _norm(x):
    return lib.sqrt_ru(tl.reduce(lib.mul_ru(x, x), 0, _up))


@triton.jit
def pack_plan(PLAN, VISIBLE, VECTORS, COEFF, ERROR, R: tl.constexpr, S: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, 64)
    if tl.load(VISIBLE + row) > S:
        base = PLAN + row * (R * 66 + 193)
        al = tl.load(base + R * 66 + d)
        ah = tl.load(base + R * 66 + 64 + d)
        # Any center is valid: certify its actual BF16 representation.
        a = ((al + ah) * 0.5).to(tl.bfloat16).to(tl.float32)
        delta = tl.maximum(tl.abs(lib.add_rd(al, -a)), tl.abs(lib.add_ru(ah, -a)))
        envelope = lib.add_ru(tl.load(base + R * 66 + 128 + d), delta)
        allowance = _norm(envelope)
        dot_error = _norm(a)
        for j in tl.static_range(R):
            q = tl.load(base + j * 64 + d)
            wl = tl.load(base + R * 64 + j)
            wh = tl.load(base + R * 65 + j)
            tl.store(VECTORS + (row * (R + 1) + j) * 64 + d, q)
            tl.store(COEFF + row * R + j, wh)
            dot_error = lib.add_ru(dot_error, lib.mul_ru(tl.maximum(tl.abs(wl), tl.abs(wh)), _norm(q)))
        # gamma_128 using a full FP32 ulp is < 2^-15. This covers
        # 64-product FP32 accumulation, including tensor-core grouping.
        allowance = lib.add_ru(allowance, lib.mul_ru(dot_error, 2.0**-15))
        ok = (tl.load(base + R * 66 + 192) != 0) & (tl.sum((~_safe(a)).to(tl.int32), 0) == 0)
        tl.store(VECTORS + (row * (R + 1) + R) * 64 + d, a)
        tl.store(ERROR + row, tl.where(ok & (allowance < float("inf")), allowance, float("inf")))


@triton.jit
def key_norm(K, NORMS, N: tl.constexpr):
    t = tl.program_id(0) * 128 + tl.arange(0, 128)
    d = tl.arange(0, 64)
    k = tl.load(K + t[:, None] * 64 + d[None, :], t[:, None] < N, 0).to(tl.float32)
    norm = lib.sqrt_ru(tl.reduce(lib.mul_ru(k, k), 1, _up))
    ok = tl.sum((~_safe(k)).to(tl.int32), 1) == 0
    tl.store(NORMS + t, tl.where(ok, norm, float("inf")), t < N)


@triton.jit
def seed_codes(SCORES, SEEDS, TOP, VISIBLE, VALID, CODES, TAU, S: tl.constexpr, CAP: tl.constexpr):
    row = tl.program_id(0)
    t = tl.arange(0, 2048)
    count = tl.load(VISIBLE + row)
    if count <= 2048:
        ids = tl.load(VALID + t, t < count, -1)
        scores = tl.full((2048,), 0, tl.bfloat16)
    else:
        selected = tl.load(TOP + row * 2048 + t)
        ids = tl.load(SEEDS + row * S + selected)
        scores = tl.load(SCORES + row * S + selected)
    tl.store(CODES + row * (CAP + 2048) + t, _code(scores, ids, ids >= 0))
    tl.store(TAU + row, tl.min(scores.to(tl.float32), 0))


@triton.jit
def complement(t, base, S: tl.constexpr):
    missing = base - S
    return tl.where(t < missing, ((t + 1) * base + tl.maximum(missing, 1) - 1) // tl.maximum(missing, 1) - 1, t + S)


@triton.jit
def screen(
    K,
    NORMS,
    VECTORS,
    COEFF,
    ERROR,
    TAU,
    BASE,
    VALID,
    VISIBLE,
    FLAGS,
    COUNTS,
    UNSAFE,
    Q: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    CAP: tl.constexpr,
    TILES: tl.constexpr,
    M: tl.constexpr,
    BOUNDS=None,
    DUMP: tl.constexpr = False,
):
    group, block = tl.program_id(0), tl.program_id(1)
    t = block * 128 + tl.arange(0, 128)
    base = tl.load(BASE + group)
    rank = complement(t, base, S)
    ids = tl.load(VALID + rank, (t < CAP) & (rank < N), -1)
    d, m = tl.arange(0, 64), tl.arange(0, M)
    L: tl.constexpr = M // 4
    row = group * 4 + m // L
    head = m % L
    vec = tl.load(
        VECTORS + (row[:, None] * (R + 1) + head[:, None]) * 64 + d[None, :],
        (head[:, None] <= R) & (row[:, None] < Q),
        0,
    )
    keys = tl.load(K + ids[None, :] * 64 + d[:, None], (ids[None, :] >= 0) & (ids[None, :] < N), 0)
    dots = tl.dot(vec, keys, input_precision="ieee")
    weight = tl.load(COEFF + row * R + head, (row < Q) & (head < R), 0)
    weight = tl.where(head == R, -1.0, weight)
    contributions = lib.mul_ru(weight[:, None], tl.maximum(dots, 0))
    # Reduce the actual head dimension once. Repeated row extraction from
    # an MMA accumulator introduces expensive layout shuffles/shared traffic.
    upper = tl.reduce(contributions.reshape((4, L, 128)), 1, _up)
    norm = tl.load(NORMS + ids, (ids >= 0) & (ids < N), float("inf"))
    qr = group * 4 + tl.arange(0, 4)
    count = tl.load(VISIBLE + qr, qr < Q, 0)
    error = tl.load(ERROR + qr, qr < Q, float("inf"))
    upper = lib.mul_ru(lib.add_ru(upper, lib.mul_ru(error[:, None], norm[None, :])), 0.125)
    unsafe = ~(tl.abs(upper) < float("inf")) | ~(norm[None, :] < float("inf"))
    upper = tl.where(unsafe, float("inf"), upper)
    if DUMP:
        tl.store(BOUNDS + qr[:, None] * CAP + t[None, :], upper, (qr[:, None] < Q) & (t[None, :] < CAP))
    threshold = tl.load(TAU + qr, qr < Q, float("inf"))
    valid = (t[None, :] < CAP) & (rank[None, :] < count[:, None]) & (count[:, None] > S) & (qr[:, None] < Q)
    live = valid & ~(upper < threshold[:, None])
    tl.store(FLAGS + qr[:, None] * CAP + t[None, :], live, (qr[:, None] < Q) & (t[None, :] < CAP))
    tl.store(COUNTS + qr * TILES + block, tl.sum(live.to(tl.int32), 1), qr < Q)
    tl.store(UNSAFE + qr * TILES + block, tl.sum((unsafe & valid).to(tl.int32), 1), qr < Q)


@triton.jit
def compact(FLAGS, PREFIX, BASE, VALID, IDS, CAP: tl.constexpr, TILES: tl.constexpr, S: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    t = block * 128 + tl.arange(0, 128)
    live = tl.load(FLAGS + row * CAP + t, t < CAP, False)
    start = tl.load(PREFIX + row * TILES + block - 1, block > 0, 0)
    dest = start + tl.cumsum(live.to(tl.int32)) - 1
    rank = complement(t, tl.load(BASE + row // 4), S)
    ids = tl.load(VALID + rank, live, 0)
    tl.store(IDS + row * CAP + dest, ids, live)


@triton.jit
def refine(Q, K, W, IDS, PREFIX, CODES, CAP: tl.constexpr, TILES: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    count = tl.load(PREFIX + row * TILES + TILES - 1)
    t = block * 128 + tl.arange(0, 128)
    if block * 128 < count:
        ids = tl.load(IDS + row * CAP + t, t < count, 0)
        h, d = tl.arange(0, 16), tl.arange(0, 64)
        q = tl.load(Q + row * 1024 + h[:, None] * 64 + d[None, :])
        w = tl.load(W + row * 16 + h)
        keys = tl.load(K + ids[None, :] * 64 + d[:, None], t[None, :] < count, 0)
        scores, _ = score_tile(q, keys, w)
        tl.store(CODES + row * (CAP + 2048) + 2048 + t, _code(scores, ids, t < count), t < CAP)


@triton.jit
def decode(CODES, OUT):
    t = tl.program_id(0) * 2048 + tl.arange(0, 2048)
    codes = tl.load(CODES + t).to(tl.uint64)
    tl.store(OUT + t, tl.where(codes != 0, (0xFFFFFFFF - (codes & 0xFFFFFFFF)).to(tl.int32), -1))


@triton.jit
def make_seeds(VALID, VISIBLE, SEEDS, BASE, N: tl.constexpr, Q: tl.constexpr, S: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    count = tl.load(VISIBLE + row)
    group = row // 4
    base = N
    for i in tl.static_range(4):
        visible = tl.load(VISIBLE + group * 4 + i, group * 4 + i < Q, 0)
        base = tl.minimum(base, tl.where(visible > S, visible, N))
    j = block * 128 + tl.arange(0, 128)
    rank = tl.where(count > S, j * base // S, j)
    ids = tl.load(VALID + rank, rank < count, -1)
    tl.store(SEEDS + row * S + j, ids)
    if (row % 4 == 0) & (block == 0):
        tl.store(BASE + group, base)


@triton.jit
def radix_select(CODES, OUT, VISIBLE, N: tl.constexpr, B: tl.constexpr, HAS_VISIBLE: tl.constexpr):
    """Exact 48-bit score/ID cutoff; no repeated sorting/merging of K
    arrays."""
    row = tl.program_id(0)
    count = tl.load(VISIBLE + row) if HAS_VISIBLE else 2049
    if count <= 2048:
        direct = tl.arange(0, 2048)
        tl.store(OUT + row * 2048 + direct, tl.load(CODES + row * N + direct))
    else:
        t = tl.arange(0, B)
        bins = tl.arange(0, 256)
        prefix = tl.full((), 0, tl.uint64)
        mask = tl.full((), 0, tl.uint64)
        remaining = 2048
        for byte in tl.static_range(5, -1, -1):
            histogram = tl.full((256,), 0, tl.int32)
            for start in range(0, N, B):
                code = tl.load(CODES + row * N + start + t, start + t < N, 0).to(tl.uint64)
                bucket = tl.where((code & mask) == prefix, (code >> (8 * byte)) & 255, 256).to(tl.int32)
                histogram += tl.histogram(bucket, 256)
            above = tl.sum(histogram, 0) - tl.cumsum(histogram)
            boundary = (above < remaining) & (above + histogram >= remaining)
            chosen = tl.sum(tl.where(boundary, bins, 0), 0)
            remaining -= tl.sum(tl.where(boundary, above, 0), 0)
            prefix |= chosen.to(tl.uint64) << (8 * byte)
            mask |= tl.full((), 255, tl.uint64) << (8 * byte)
        written = 0
        for start in range(0, N, B):
            code = tl.load(CODES + row * N + start + t, start + t < N, 0).to(tl.uint64)
            live = (code >= prefix) & (code != 0) & (start + t < N)
            destination = written + tl.cumsum(live.to(tl.int32)) - 1
            tl.store(OUT + row * 2048 + destination, code, live & (destination < 2048))
            written += tl.sum(live.to(tl.int32), 0)


def select_codes(codes: "torch.Tensor", visible: "torch.Tensor | None" = None) -> "torch.Tensor":
    """Select exact 48-bit BF16 score/original-ID codes; padding stays zero."""
    import torch

    rows, capacity = codes.shape
    result = torch.zeros((rows, 2048), device=codes.device, dtype=torch.int64)
    radix_select[(rows,)](
        codes, result, visible, capacity, min(2048, triton.next_power_of_2(capacity)), visible is not None, num_warps=4
    )
    return result
