# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Conservative BF16 DoSA primitives. No persistent state or GPU/CPU decisions.

Screening uses directed FP32 arithmetic, not a tolerance fitted to test data.
Refinement uses the same fixed MMA tile as DSA. The streaming implementation
has bounded scratch space even when every key survives (including all ties).
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

from .indexer_scores import score_tile


@triton.jit
def _up(a, b):
    return lib.add_ru(a, b)


@triton.jit
def _down(a, b):
    return lib.add_rd(a, b)


@triton.jit
def _safe(x):
    # With BF16 inputs this also bounds the smallest nonzero quantum of
    # dot, scaled dot, weighted dot and final sum above FP32/BF16 underflow.
    a = tl.abs(x.to(tl.float32))
    return (a == 0) | ((a >= 2.0**-32) & (a <= 2.0**16))


@triton.jit
def _code(scores, ids, valid):
    bits = scores.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    # Treat -0 and +0 as the same score, as the original top-k does.
    bits = tl.where((bits & 0x7FFF) == 0, 0, bits)
    ordered = tl.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits ^ 0x8000)
    return tl.where(valid, (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - ids.to(tl.uint32)), 0)


@triton.jit
def _merge(best, extra):
    # First 2048 codes of a descending merge. Original ID is in the code,
    # so neither compaction order nor score ties can change membership.
    combined = tl.reshape(tl.join(best, extra), (4096,))
    ordered = tl.sort(combined, descending=True)
    return tl.gather(ordered, tl.arange(0, 2048), 0)


@triton.jit
def seed_scores(Q, K, W, SEEDS, VISIBLE, OUT, QSTR: tl.constexpr, WSTR: tl.constexpr, S: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    t = block * 128 + tl.arange(0, 128)
    ids = tl.load(SEEDS + row * S + t)
    count = tl.load(VISIBLE + row)
    h, d = tl.arange(0, 16), tl.arange(0, 64)
    result = tl.full((128,), float("-inf"), tl.bfloat16)
    if (count > 2048) & (block * 128 < count):
        q = tl.load(Q + row * QSTR + h[:, None] * 64 + d[None, :])
        w = tl.load(W + row * WSTR + h)
        k = tl.load(K + ids[None, :] * 64 + d[:, None], ids[None, :] >= 0, 0)
        result, _ = score_tile(q, k, w)
    tl.store(OUT + row * S + t, tl.where(ids >= 0, result, float("-inf")))


@triton.jit
def seed_topk(SCORES, SEEDS, VISIBLE, OUTPUT, S: tl.constexpr):
    row = tl.program_id(0)
    if tl.load(VISIBLE + row) > 2048:
        j = tl.arange(0, S)
        scores = tl.load(SCORES + row * S + j)
        ids = tl.load(SEEDS + row * S + j)
        codes = tl.sort(_code(scores, j, ids >= 0), descending=True)
        codes = tl.gather(codes, tl.arange(0, 2048), 0)
        tl.store(OUTPUT + row * 2048 + tl.arange(0, 2048), (0xFFFFFFFF - (codes & 0xFFFFFFFF)).to(tl.int32))


@triton.jit
def prepare(
    Q,
    W,
    K,
    SEEDS,
    VISIBLE,
    PLAN,
    QSTR: tl.constexpr,
    WSTR: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    DIAG=None,
    DUMP: tl.constexpr = False,
    HEAD_RANKS=None,
    OVERRIDE: tl.constexpr = False,
):
    if tl.load(VISIBLE + tl.program_id(0)) > S:
        _prepare(Q, W, K, SEEDS, PLAN, QSTR, WSTR, S, R, DIAG, DUMP, HEAD_RANKS, OVERRIDE)


@triton.jit
def _norm_up(x):
    return lib.sqrt_ru(tl.reduce(lib.mul_ru(x, x), 0, _up))


@triton.jit
def _joint_allowance(lo, hi):
    """Enclose Eq. (7) using signed weighted-residual intervals.

    Each Gram interval encloses the exact residual dot product. Absolute values
    enter only after the signed reduction, preserving cancellation.
    """
    h = tl.arange(0, 16)
    total_lo = tl.reduce(lo, 0, _down)
    total_hi = tl.reduce(hi, 0, _up)
    center_norm = _norm_up(tl.maximum(tl.abs(total_lo), tl.abs(total_hi)))
    gram_sum = tl.full((), 0, tl.float32)
    for i in range(16):
        a = tl.sum(tl.where((h == i)[:, None], lo, 0), 0)
        b = tl.sum(tl.where((h == i)[:, None], hi, 0), 0)
        lower = tl.minimum(
            tl.minimum(lib.mul_rd(lo, a), lib.mul_rd(lo, b)),
            tl.minimum(lib.mul_rd(hi, a), lib.mul_rd(hi, b)),
        )
        upper = tl.maximum(
            tl.maximum(lib.mul_ru(lo, a), lib.mul_ru(lo, b)),
            tl.maximum(lib.mul_ru(hi, a), lib.mul_ru(hi, b)),
        )
        dot_lo = tl.reduce(lower, 1, _down)
        dot_hi = tl.reduce(upper, 1, _up)
        absolute = tl.maximum(tl.abs(dot_lo), tl.abs(dot_hi))
        gram_sum = lib.add_ru(gram_sum, tl.reduce(absolute, 0, _up))
    return lib.mul_ru(lib.add_ru(center_norm, lib.sqrt_ru(gram_sum)), 0.5)


@triton.jit
def _prepare(
    Q,
    W,
    K,
    SEEDS,
    PLAN,
    QSTR: tl.constexpr,
    WSTR: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    DIAG=None,
    DUMP: tl.constexpr = False,
    HEAD_RANKS=None,
    OVERRIDE: tl.constexpr = False,
):
    """Nonnegative retained-head combination; joint residual allowance.

    PLAN stores retained vectors, folded coefficient intervals, aggregate
    intervals, a reserved coordinate envelope, and a scalar error allowance.
    Infinite allowance marks an ineligible certificate. Projection quality
    affects pruning only: any nonnegative beta is safe.
    """
    row = tl.program_id(0)
    h, d = tl.arange(0, 16), tl.arange(0, 64)
    q = tl.load(Q + row * QSTR + h[:, None] * 64 + d[None, :])
    w = tl.load(W + row * WSTR + h).to(tl.float32)
    x = q.to(tl.float32)
    ids = tl.load(SEEDS + row * S + tl.arange(0, 128))
    k = tl.load(K + ids[None, :] * 64 + d[:, None], ids[None, :] >= 0, 0)
    dots = tl.dot(q, k, input_precision="ieee")
    freq = tl.sum(((dots > 0) & (ids[None, :] >= 0)).to(tl.float32), 1) / tl.maximum(
        tl.sum((ids >= 0).to(tl.float32), 0), 1
    )
    norm = tl.sqrt(tl.sum(x * x, 1))
    priority = tl.abs(w) * norm * tl.where(w > 0, 1.0, tl.minimum(freq, 1.0 - freq))
    positive = w > 0
    better = (positive[None, :] & ~positive[:, None]) | (
        (positive[None, :] == positive[:, None])
        & (
            (priority[None, :] > priority[:, None])
            | ((priority[None, :] == priority[:, None]) & (h[None, :] < h[:, None]))
        )
    )
    rank = tl.sum(better.to(tl.int32), 1)
    if OVERRIDE:
        # Diagnostic-only per-query permutation; callers validate it on CPU.
        rank = tl.load(HEAD_RANKS + row * 16 + h)
    retained = rank < R
    omitted = positive & ~retained
    best = tl.full((16,), -1.0, tl.float32)
    anchor_id = tl.full((16,), 0, tl.int32)
    beta = tl.full((16,), 0, tl.float32)
    anchor = tl.full((16, 64), 0, tl.float32)
    # Selection/projection need not be directed: certify the residual of the
    # actual finite nonnegative coefficient, including any selection rounding.
    for j in tl.static_range(R):
        vector = tl.sum(tl.where((rank == j)[:, None], x, 0), 0)
        denominator = tl.sum(vector * vector, 0)
        dot = tl.maximum(tl.sum(x * vector[None, :], 1), 0)
        quality = dot / tl.sqrt(tl.maximum(denominator, 2.0**-64))
        take = quality > best
        projection = tl.minimum(dot / tl.maximum(denominator, 2.0**-64), 65536.0)
        beta = tl.where(take, projection, beta)
        anchor = tl.where(take[:, None], vector[None, :], anchor)
        anchor_id = tl.where(take, j, anchor_id)
        best = tl.maximum(best, quality)
    beta = tl.where(omitted, beta, 0)
    # Paper Eq. (4): one best retained anchor per omitted positive head.
    # Certify the actual finite coefficient rather than assuming exact fitting.
    columns = tl.arange(0, 16)
    single_coeff = tl.where(columns[None, :] == anchor_id[:, None], beta[:, None], 0)
    coefficients = single_coeff
    projection_lo = tl.full((16, 64), 0, tl.float32)
    projection_hi = tl.full((16, 64), 0, tl.float32)
    for j in tl.static_range(R):
        vector = tl.sum(tl.where((rank == j)[:, None], x, 0), 0)
        coefficient = tl.sum(tl.where(columns[None, :] == j, coefficients, 0), 1)
        projection_lo = lib.add_rd(projection_lo, lib.mul_rd(coefficient[:, None], vector[None, :]))
        projection_hi = lib.add_ru(projection_hi, lib.mul_ru(coefficient[:, None], vector[None, :]))
    residual_lo = lib.mul_rd(w[:, None], lib.add_rd(x, -projection_hi))
    residual_hi = lib.mul_ru(w[:, None], lib.add_ru(x, -projection_lo))
    residual_lo = tl.where(omitted[:, None], residual_lo, 0)
    residual_hi = tl.where(omitted[:, None], residual_hi, 0)
    residual = tl.maximum(tl.abs(residual_lo), tl.abs(residual_hi))
    if DUMP:
        # Per-head coefficients, single-anchor/certified weighted residual norm
        # enclosures, original weighted norm, and unchanged retained rank.
        single_lo = lib.mul_rd(w[:, None], lib.add_rd(x, -lib.mul_ru(beta[:, None], anchor)))
        single_hi = lib.mul_ru(w[:, None], lib.add_ru(x, -lib.mul_rd(beta[:, None], anchor)))
        single_abs = tl.where(omitted[:, None], tl.maximum(tl.abs(single_lo), tl.abs(single_hi)), 0)
        single_norm = lib.sqrt_ru(tl.reduce(lib.mul_ru(single_abs, single_abs), 1, _up))
        final_norm = lib.sqrt_ru(tl.reduce(lib.mul_ru(residual, residual), 1, _up))
        tl.store(DIAG + row * 320 + h[:, None] * 20 + columns[None, :], coefficients)
        tl.store(DIAG + row * 320 + h * 20 + 16, single_norm)
        tl.store(DIAG + row * 320 + h * 20 + 17, final_norm)
        tl.store(DIAG + row * 320 + h * 20 + 18, tl.abs(w) * norm)
        tl.store(DIAG + row * 320 + h * 20 + 19, rank.to(tl.float32))
    envelope = tl.reduce(residual, 0, _up)
    # Preserve the original BF16 reference-arithmetic allowance (2^-6).
    arithmetic = tl.reduce(lib.mul_ru(tl.abs(w[:, None]), tl.abs(x)), 0, _up)
    arithmetic = lib.mul_ru(arithmetic, 2.0**-6)
    coordinate = _norm_up(lib.add_ru(envelope, arithmetic))
    joint = lib.add_ru(_joint_allowance(residual_lo, residual_hi), _norm_up(arithmetic))
    joint = tl.where(tl.abs(joint) < float("inf"), joint, float("inf"))
    allowance = tl.minimum(coordinate, joint)
    base = PLAN + row * (R * 66 + 193)
    for j in tl.static_range(R):
        vector = tl.sum(tl.where((rank == j)[:, None], x, 0), 0)
        weight = tl.sum(tl.where(rank == j, w, 0), 0)
        tl.store(base + j * 64 + d, vector)
        coefficient = tl.sum(tl.where(columns[None, :] == j, coefficients, 0), 1)
        folded_lo = tl.reduce(lib.mul_rd(tl.where(omitted, w, 0), coefficient), 0, _down)
        folded_hi = tl.reduce(lib.mul_ru(tl.where(omitted, w, 0), coefficient), 0, _up)
        tl.store(base + R * 64 + j, lib.add_rd(weight, folded_lo))
        tl.store(base + R * 65 + j, lib.add_ru(weight, folded_hi))
    negative = (w < 0) & ~retained & (freq >= 0.5)
    magnitude = tl.where(negative, -w, 0)
    aggregate_lo = tl.reduce(lib.mul_rd(magnitude[:, None], x), 0, _down)
    aggregate_hi = tl.reduce(lib.mul_ru(magnitude[:, None], x), 0, _up)
    tl.store(base + R * 66 + d, aggregate_lo)
    tl.store(base + R * 66 + 64 + d, aggregate_hi)
    tl.store(base + R * 66 + 128 + d, tl.full((64,), 0, tl.float32))
    ok = tl.sum((~_safe(q)).to(tl.int32)) + tl.sum((~_safe(w)).to(tl.int32)) == 0
    tl.store(base + R * 66 + 192, tl.where(ok, allowance, float("inf")))


@triton.jit
def _bound(K, ids, PLAN, R: tl.constexpr):
    d = tl.arange(0, 64)
    k = tl.load(K + ids[:, None] * 64 + d[None, :], ids[:, None] >= 0, 0).to(tl.float32)
    upper = tl.full((128,), 0, tl.float32)
    for j in tl.static_range(R):
        q = tl.load(PLAN + j * 64 + d)
        lo = tl.maximum(tl.reduce(lib.mul_rd(k, q[None, :]), 1, _down), 0)
        hi = tl.maximum(tl.reduce(lib.mul_ru(k, q[None, :]), 1, _up), 0)
        wl = tl.load(PLAN + R * 64 + j)
        wh = tl.load(PLAN + R * 65 + j)
        contribution = tl.maximum(lib.mul_ru(wh, hi), lib.mul_ru(wh, lo))
        contribution = tl.maximum(contribution, tl.maximum(lib.mul_ru(wl, lo), lib.mul_ru(wl, hi)))
        upper = lib.add_ru(upper, contribution)
    al = tl.load(PLAN + R * 66 + d)
    ah = tl.load(PLAN + R * 66 + 64 + d)
    neg_lo = tl.reduce(tl.minimum(lib.mul_rd(k, al[None, :]), lib.mul_rd(k, ah[None, :])), 1, _down)
    envelope = tl.load(PLAN + R * 66 + 128 + d)
    allowance = tl.reduce(lib.mul_ru(tl.abs(k), envelope[None, :]), 1, _up)
    scalar = tl.load(PLAN + R * 66 + 192)
    norm = lib.sqrt_ru(tl.reduce(lib.mul_ru(k, k), 1, _up))
    allowance = lib.add_ru(allowance, lib.mul_ru(scalar, norm))
    upper = lib.mul_ru(lib.add_ru(lib.add_ru(upper, allowance), -tl.maximum(neg_lo, 0)), 0.125)
    ok = (tl.sum((~_safe(k)).to(tl.int32), 1) == 0) & (scalar < float("inf"))
    # Includes NaN/overflow in either preparation or evaluation.
    return tl.where(ok & (tl.abs(upper) < float("inf")), upper, float("inf"))


@triton.jit
def select_stream(
    Q,
    K,
    W,
    PLAN,
    VALID_IDS,
    VISIBLE,
    SEEDS,
    SEED_SCORES,
    SEED_TOP,
    OUTPUT,
    STATS,
    QSTR: tl.constexpr,
    WSTR: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
):
    row = tl.program_id(0)
    a = tl.arange(0, 2048)
    count = tl.load(VISIBLE + row)
    if count <= 2048:
        ids = tl.load(VALID_IDS + a, a < count, 0)
        tl.store(OUTPUT + row * 2048 + a, tl.where(a < count, ids, -1))
        tl.store(STATS + row * 2, 0)
        tl.store(STATS + row * 2 + 1, 0)
    else:
        selected = tl.load(SEED_TOP + row * 2048 + a)
        seed_ids = tl.load(SEEDS + row * S + selected)
        seed_scores = tl.load(SEED_SCORES + row * S + selected)
        best = _code(seed_scores, seed_ids, seed_ids >= 0)
        tau = tl.min(seed_scores.to(tl.float32), 0)
        plan = PLAN + row * (R * 66 + 193)
        h, d, t = tl.arange(0, 16), tl.arange(0, 64), tl.arange(0, 128)
        q = tl.load(Q + row * QSTR + h[:, None] * 64 + d[None, :])
        w = tl.load(W + row * WSTR + h)
        survivors, unsafe = 0, 0
        pending = tl.full((2048,), 0, tl.uint64)
        pending_count = 0
        for start in range(0, tl.where(count > S, count, 0), 128):
            rank = start + t
            ids = tl.load(VALID_IDS + rank, rank < count, -1)
            # Seeds are floor(j*count/S) if count>S, otherwise every key.
            j = (rank * S + count - 1) // count
            is_seed = (count <= S) | ((j < S) & ((j * count) // S == rank))
            upper = _bound(K, ids, plan, R)
            live = (rank < count) & ~is_seed & ~(upper < tau)
            survivors += tl.sum(live.to(tl.int32), 0)
            pending_count += tl.sum(live.to(tl.int32), 0)
            unsafe += tl.sum(((upper == float("inf")) & (rank < count) & ~is_seed).to(tl.int32), 0)
            if tl.sum(live.to(tl.int32), 0) > 0:
                # Local compaction has capacity 128 by construction. Padded
                # MMA columns load zeros, never a rejected key.
                compact = tl.sort(tl.where(live, ids, 0x7FFFFFFF), descending=False)
                valid = compact != 0x7FFFFFFF
                keys = tl.load(K + compact[None, :] * 64 + d[:, None], valid[None, :], 0)
                scores, _ = score_tile(q, keys, w)
                codes = _code(scores, compact, valid)
                extra = tl.gather(codes, a % 128, 0)
                pending = tl.where(a // 128 == (start // 128) % 16, extra, pending)
            # Amortize selection over sixteen scan tiles. Capacity is still
            # bounded: each of the 16 disjoint slices receives at most 128 IDs.
            if (start % 2048 == 1920) | (start + 128 >= count):
                if pending_count > 0:
                    best = _merge(best, pending)
                pending = tl.full((2048,), 0, tl.uint64)
                pending_count = 0
        ids = (0xFFFFFFFF - (best & 0xFFFFFFFF)).to(tl.int32)
        tl.store(OUTPUT + row * 2048 + a, ids)
        tl.store(STATS + row * 2, survivors)
        tl.store(STATS + row * 2 + 1, unsafe)
