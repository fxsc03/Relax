# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye BF16 DoSA requires CUDA")


def _uniform_select(*args, **kwargs):
    """Keep the original uniform-seed regression matrix as an explicit
    ablation."""
    from relax.models.keye_vl2.dosa import dosa_select

    kwargs.setdefault("seed_policy", "uniform")
    kwargs.setdefault("seed_size", 4096)
    return dosa_select(*args, **kwargs)


def test_backend_config_and_actual_freeze() -> None:
    from relax.models.keye_vl2.dosa import backend_config, frozen_indexer_enabled

    module = torch.nn.Linear(2, 2)
    config = SimpleNamespace(keye_dsa_backend="dosa")
    assert backend_config(SimpleNamespace()) == ("dsa", 4)
    assert backend_config(config) == ("dosa", 4)
    with pytest.raises(ValueError):
        backend_config(SimpleNamespace(keye_dsa_backend="typo"))
    with torch.no_grad():
        assert not frozen_indexer_enabled(config, module)
    module.requires_grad_(False)
    assert frozen_indexer_enabled(config, module)
    assert not frozen_indexer_enabled(SimpleNamespace(), module)
    assert _uniform_select(torch.empty(1, 16, 64), torch.empty(1, 1, 64), torch.empty(1, 16), torch.zeros(1)) is None


def _inputs(n: int, rows: int = 4) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(293)
    q = torch.randn(rows, 16, 128, device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    k = torch.randn(n, 1, 128, device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    w = torch.randn(rows, 16, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([-1, min(2047, n - 1), n - 9, n - 1], device="cuda")[:rows]
    valid = torch.ones(n, device="cuda", dtype=torch.bool)
    valid[37::113] = False
    return q, k, w, offsets, valid


def test_provider_consumes_model_specific_yaml_fields() -> None:
    from relax.models.keye_vl2.provider import KeyeModelProvider

    provider = SimpleNamespace(context_parallel_size=1, position_embedding_type="rope")
    args = SimpleNamespace(multimodal_keys=[], keye_dsa_backend="dosa", keye_dosa_heads=7)
    KeyeModelProvider.configure_runtime(provider, args)
    assert (provider.keye_dsa_backend, provider.keye_dosa_heads) == ("dosa", 7)
    KeyeModelProvider.configure_runtime(provider, SimpleNamespace(multimodal_keys=[]))
    assert (provider.keye_dsa_backend, provider.keye_dosa_heads) == ("dsa", 4)


@gpu
def test_unsupported_inputs_request_original_path() -> None:

    q, k, w, offsets, valid = _inputs(4097)
    assert _uniform_select(q.float(), k.float(), w.float(), offsets) is None
    assert _uniform_select(q[:, :8], k, w[:, :8], offsets) is None
    assert _uniform_select(q, k, w, offsets.cpu()) is None
    assert _uniform_select(q, k, w, offsets, valid.cpu()) is None


def _assert_selection(q, k, w, offsets, valid, heads=4):
    from relax.models.keye_vl2.indexer_scores import dense_indexer_scores
    from relax.models.keye_vl2.sparse_attention import compact_indices
    from relax.models.keye_vl2.topk import dsa_topk_indices

    stats = {}
    actual = _uniform_select(q, k, w, offsets, valid, heads=heads, stats=stats)
    causal = (torch.arange(len(k), device=q.device)[None] <= offsets[:, None]) & valid[None]
    scores = dense_indexer_scores(q, k, w).masked_fill(~causal, torch.finfo(q.dtype).min)
    expected = dsa_topk_indices(scores)
    if actual is None:
        assert 2048 < len(k) <= 4096  # Measured short-prefix fallback.
        return stats
    actual_compact = compact_indices(actual.clamp_min(0), actual >= 0)
    expected_compact = compact_indices(expected, causal.gather(1, expected.long()))
    for a, b in zip(actual_compact, expected_compact):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    return stats


@gpu
@pytest.mark.parametrize("length", [2047, 2048, 2049, 4096, 4097, 8193])
def test_exact_compact_order_causal_and_padding(length: int) -> None:
    _assert_selection(*_inputs(length))


@gpu
@pytest.mark.parametrize(
    "kind", ["zero", "ties", "negative", "unsafe", "nonfinite", "nan_query", "nan_weight", "inf_key", "inf_weight"]
)
def test_signed_rounding_ties_and_conservative_recovery(kind: str) -> None:
    q, k, w, offsets, valid = _inputs(8193)
    if kind == "zero":
        q.zero_()
    elif kind == "ties":
        k[:] = k[0].clone()
    elif kind == "negative":
        w.copy_(-w.abs())
    elif kind == "nonfinite":
        k[5000] = float("nan")
    elif kind == "nan_query":
        q[:, 0, 0] = float("nan")
    elif kind == "nan_weight":
        w[:, 0] = float("nan")
    elif kind == "inf_key":
        k[5001] = float("inf")
    elif kind == "inf_weight":
        w[:, 0] = float("inf")
    else:
        q[:, :, 0] = 2.0**-40  # Cannot use the normal-range numerical certificate.
    stats = _assert_selection(q, k, w, offsets, valid)
    if kind == "unsafe":
        assert (stats["unbounded"][2:] > 0).all()


@gpu
@pytest.mark.parametrize("heads", [1, 4, 15])
def test_actual_pruning_and_original_ids(heads: int) -> None:
    q, k, w, offsets, valid = _inputs(8193)
    q.zero_()
    q[:, :, 0] = 1
    k.zero_()
    k[:, 0, 0] = torch.linspace(0.125, 4, len(k), device=k.device)
    w.fill_(1)
    stats = _assert_selection(q, k, w, offsets, valid, heads)
    remaining = stats["visible"][-1] - stats["seed_count"][-1]
    assert 0 < stats["survivors"][-1] < remaining
    assert stats["exact_score_rejected"][-1] > 0
    assert stats["selection_candidates"][-1] == 2048 + stats["survivors"][-1] - stats["exact_score_rejected"][-1]


@gpu
def test_reference_and_bf16_midpoint() -> None:
    from relax.models.keye_vl2.dosa_kernels import seed_scores
    from relax.models.keye_vl2.dosa_reference import reference_bound, reference_select
    from relax.models.keye_vl2.indexer_scores import dense_indexer_scores
    from tests.models.keye_vl2.test_indexer_scores import _rounding_inputs

    q, k, w = _rounding_inputs()
    q, w = q[120:121], w[120:121]
    extended = k.new_zeros((8193, 1, 64))
    extended[5001] = k[503]  # Odd ID is outside the uniform even-ID seed set.
    k = extended
    offsets = torch.tensor([len(k) - 1], device=q.device)
    valid = torch.ones(len(k), device=q.device, dtype=torch.bool)
    _assert_selection(q, k, w, offsets, valid)
    bounds = reference_bound(q[0], k, w[0])
    full_scores = dense_indexer_scores(q, k, w)
    assert (bounds >= full_scores[0].double()).all()
    # Non-contiguous gathering must preserve the midpoint score itself, not
    # merely select the same easy winner after a numerically different dot.
    gathered = (2 * torch.arange(4096, device=q.device)[None] + 1).int()
    actual_scores = q.new_empty((1, 4096))
    seed_scores[(1, 32)](
        q,
        k,
        w,
        gathered,
        offsets + 1,
        actual_scores,
        q.stride(0),
        w.stride(0),
        4096,
        num_warps=4,
        enable_fp_fusion=False,
    )
    torch.testing.assert_close(actual_scores, full_scores.gather(1, gathered.long()), atol=0, rtol=0)
    chosen = reference_select(q, k, w, offsets)
    assert 5001 in chosen[0]


@gpu
@pytest.mark.parametrize("heads", [1, 4, 15])
def test_numerical_certificate_dominates_original_scorer(heads: int) -> None:
    from relax.models.keye_vl2.dosa_kernels import _bound, prepare
    from relax.models.keye_vl2.indexer_scores import dense_indexer_scores

    @triton.jit
    def bounds_for_test(K, PLAN, OUT, N: tl.constexpr, R: tl.constexpr):
        row, block = tl.program_id(0), tl.program_id(1)
        ids = block * 128 + tl.arange(0, 128)
        bound = _bound(K, tl.where(ids < N, ids, -1), PLAN + row * (R * 66 + 193), R)
        tl.store(OUT + row * N + ids, bound, ids < N)

    q, k, w, _, _ = _inputs(8193)
    q, k, w = q.contiguous(), k.contiguous(), w.contiguous()
    # Mix signs and wide exponents, including the admitted numerical range.
    q[1].mul_(2.0**-16)
    k[1::3].mul_(2.0**16)
    w[1].mul_(2.0**-16)
    seeds = (torch.arange(4096, device=q.device)[None].expand(4, -1) * 2).int().contiguous()
    visible = torch.full((4,), len(k), device=q.device, dtype=torch.int32)
    plan = torch.empty(4, heads * 66 + 193, device=q.device, dtype=torch.float32)
    upper = torch.empty(4, len(k), device=q.device, dtype=torch.float32)
    prepare[(4,)](
        q, w, k, seeds, visible, plan, q.stride(0), w.stride(0), 4096, heads, num_warps=4, enable_fp_fusion=False
    )
    bounds_for_test[(4, triton.cdiv(len(k), 128))](k, plan, upper, len(k), heads, num_warps=4, enable_fp_fusion=False)
    assert (upper >= dense_indexer_scores(q, k, w).float()).all()


@gpu
def test_frozen_checkpoint_output_and_gradients(monkeypatch) -> None:
    from copy import deepcopy

    from torch.utils.checkpoint import checkpoint

    from relax.models.keye_vl2 import attention as implementation
    from tests.models.keye_vl2.test_keye_vl2_attention import _attention, _inputs

    original_select = implementation.dosa_select
    selections = []

    def record(*args, **kwargs):
        # This4K regression explicitly exercises the non-fallback seed path.
        kwargs["seed_size"] = 4096
        result = original_select(*args, **kwargs)
        if result is not None and len(args[1]) > 4096:
            selections.append((torch.is_grad_enabled(), result.clone()))
        return result

    monkeypatch.setattr(implementation, "dosa_select", record)

    baseline = _attention()
    baseline.sa_indexer.requires_grad_(False)
    optimized = deepcopy(baseline)
    optimized.config.keye_dsa_backend = "dosa"
    inputs = _inputs(4113)
    clones = [x.detach().clone().requires_grad_() for x in inputs]
    packed = SimpleNamespace(cu_seqlens_q_cpu=[0, 4113], valid_seqlens_q_cpu=[4101])
    expected = baseline(*inputs[:3], hidden_states=inputs[3], packed_seq_params=packed)
    actual = checkpoint(
        lambda q, k, v, h: optimized(q, k, v, hidden_states=h, packed_seq_params=packed), *clones, use_reentrant=True
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    assert [grad for grad, _ in selections] == [False, True]
    torch.testing.assert_close(selections[0][1], selections[1][1], atol=0, rtol=0)
    for a, b in zip(inputs[:3], clones[:3]):
        # The unchanged fused SDPA backward uses parallel reductions. The
        # selected support and forward above remain bitwise identical.
        torch.testing.assert_close(a.grad, b.grad, atol=1e-9, rtol=1e-5)
    assert all(p.grad is None for p in optimized.sa_indexer.parameters())


@gpu
def test_trainable_checkpoint_retains_auxiliary(monkeypatch) -> None:
    import tests.models.keye_vl2.test_keye_vl2_attention as original
    from tests.models.keye_vl2.test_keye_vl2_attention import (
        _attention,
        test_keye_sparse_recompute_has_indexer_gradients_and_no_future_leak,
    )

    def attention():
        model = _attention()
        model.config.keye_dsa_backend = "dosa"
        return model

    monkeypatch.setattr(original, "_attention", attention)
    monkeypatch.setattr(
        "relax.models.keye_vl2.attention.dosa_select",
        lambda *args, **kwargs: pytest.fail("Trainable indexer entered DoSA"),
    )
    test_keye_sparse_recompute_has_indexer_gradients_and_no_future_leak()


@gpu
@pytest.mark.parametrize("prefix,rows", [(0, 7), (0, 4113), (8192, 7)])
def test_sglang_prefill_extend_with_physical_slot_permutation(monkeypatch, prefix: int, rows: int) -> None:
    from relax.models.keye_vl2.sglang import model

    torch.manual_seed(887)
    length = prefix + rows
    attention = model.KeyeAttention.__new__(model.KeyeAttention)
    torch.nn.Module.__init__(attention)
    attention.qkv_proj = lambda h: (h, None)
    attention.q_size, attention.kv_size = 256, 128
    attention.num_heads, attention.num_kv_heads, attention.head_dim = 2, 1, 128
    attention.q_norm = attention.k_norm = torch.nn.Identity()
    attention.rope_theta = 1e7
    attention.keye_multimodal_enabled = True
    attention.keye_dosa_heads = 4
    attention.sa_indexer = SimpleNamespace(
        q_proj=lambda h: torch.cat((h.repeat(1, 2), h[:, :16]), dim=-1),
        k_proj=lambda h: h[:, :64],
        q_norm=torch.nn.Identity(),
        k_norm=torch.nn.Identity(),
    )
    attention.attn = SimpleNamespace(layer_id=0)
    attention.o_proj = lambda h: (h, None)
    keys = torch.randn(length, 1, 128, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    index = torch.randn(length, 64, device="cuda", dtype=torch.bfloat16)
    table = torch.randperm(length, device="cuda")

    def write(layer, location, k, v):
        keys[location.loc] = k
        values[location.loc] = v

    pool = SimpleNamespace(
        set_kv_buffer=write,
        keye_index_cache={0: index},
        get_key_buffer=lambda layer: keys,
        get_value_buffer=lambda layer: values,
    )
    monkeypatch.setattr(model, "get_token_to_kv_pool", lambda: pool)
    monkeypatch.setattr(model, "get_req_to_token_pool", lambda: SimpleNamespace(req_to_token=table[None]))
    batch = SimpleNamespace(
        out_cache_loc=table[-rows:],
        rids=["request-a"],
        req_pool_indices=torch.tensor([0], device="cuda"),
        seq_lens_cpu=[length],
        extend_seq_lens_cpu=[rows],
        mrope_positions=torch.zeros(3, rows, device="cuda", dtype=torch.long),
        forward_mode=SimpleNamespace(is_decode=lambda: False, is_extend=lambda: True),
    )
    positions = torch.arange(prefix, length, device="cuda")
    hidden = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
    attention.keye_dsa_backend = "dsa"
    expected = attention(positions, hidden, batch)
    attention.keye_dsa_backend = "dosa"
    supplied_history = []
    original_select = model.dosa_select

    def observe(*args, **kwargs):
        supplied_history.append(kwargs["seed_state"].selected is not None)
        return original_select(*args, **kwargs)

    monkeypatch.setattr(model, "dosa_select", observe)
    actual = attention(positions, hidden, batch)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert supplied_history[0] is False
    supplied_history.clear()
    actual = attention(positions, hidden, batch)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert supplied_history[0] is (prefix > 0)
    batch.rids = ["request-b"]
    supplied_history.clear()
    actual = attention(positions, hidden, batch)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert supplied_history[0] is False  # No witnesses from another request.
    # The sidecar is updated even when <=K keys makes scoring unnecessary.
    assert torch.isfinite(pool.keye_index_cache[0][table[-rows:]]).all()


@gpu
@pytest.mark.parametrize("backend", ["dsa", "dosa"])
def test_sglang_constructor_copies_config_to_actual_attention(monkeypatch, backend: str) -> None:
    from relax.models.keye_vl2.sglang import model

    def language_init(self, config, *args):
        torch.nn.Module.__init__(self)
        self.lm_head = torch.nn.Linear(2, 2)
        self.model = torch.nn.Module()
        self.model.norm = torch.nn.Identity()
        layer = torch.nn.Module()
        layer.input_layernorm = layer.post_attention_layernorm = torch.nn.Identity()
        layer.mlp = torch.nn.Module()
        layer.mlp.gate = torch.nn.Linear(2, 2)
        layer.mlp.experts = torch.nn.Module()
        layer.mlp.topk = SimpleNamespace(topk_config=SimpleNamespace())
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_norm = layer.self_attn.k_norm = torch.nn.Identity()
        layer.self_attn.qkv_proj = torch.nn.Linear(2, 2)
        layer.self_attn.o_proj = torch.nn.Linear(2, 2)
        self.model.layers = torch.nn.ModuleList([layer])

    monkeypatch.setattr(model.Qwen3MoeForCausalLM, "__init__", language_init)
    monkeypatch.setattr(model.KeyeRMSNorm, "convert", lambda norm: None)
    monkeypatch.setattr(model, "process_model_config", lambda: SimpleNamespace(is_multimodal=False))
    monkeypatch.setattr(model, "get_parallel", lambda: SimpleNamespace(moe_tp_size=1, attn_tp_size=1))
    config = SimpleNamespace(
        hidden_size=32, rms_norm_eps=1e-6, rope_theta=1e7, keye_dsa_backend=backend, keye_dosa_heads=7
    )
    instance = model.KeyeVL2MoeForConditionalGeneration(config)
    attention = instance.model.layers[0].self_attn
    assert isinstance(attention, model.KeyeAttention)
    assert (attention.keye_dsa_backend, attention.keye_dosa_heads) == (backend, 7)


@gpu
@pytest.mark.parametrize("heads", [1, 4, 15])
def test_grouped_tensorcore_certificate(monkeypatch, heads: int) -> None:
    from relax.models.keye_vl2 import dosa_parallel
    from relax.models.keye_vl2.indexer_scores import dense_indexer_scores

    q, k, w, offsets, valid = _inputs(8193)
    offsets[:] = len(k) - 1
    valid[:] = True
    q[1].mul_(2.0**-16)
    k[1::3].mul_(2.0**16)
    w[1].mul_(2.0**-16)
    upper = torch.empty((len(q), len(k) - 4096), device=q.device)
    original = dosa_parallel.screen

    class Capture:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                return original[grid](*args, BOUNDS=upper, DUMP=True, **kwargs)

            return launch

    monkeypatch.setattr(dosa_parallel, "screen", Capture())
    _uniform_select(q, k, w, offsets, valid, heads=heads)
    seeds = torch.arange(4096, device=q.device) * len(k) // 4096
    candidate = torch.ones(len(k), device=q.device, dtype=torch.bool)
    candidate[seeds] = False
    exact = dense_indexer_scores(q, k, w)[:, candidate].float()
    assert (upper >= exact).all()


@gpu
@pytest.mark.parametrize("seed_size", [2048, 4096, 8192])
def test_cross_tile_packing_full_capacity_and_seed_sizes(seed_size: int) -> None:
    from relax.models.keye_vl2.sparse_attention import compact_indices

    q, k, w, _, _ = _inputs(16387)
    q.zero_()  # All ties: every non-seed key must survive, never truncate.
    offsets = torch.full((4,), len(k) - 1, device=q.device)
    stats = {}
    selected = _uniform_select(q, k, w, offsets, seed_size=seed_size, stats=stats)
    ordered, counts = compact_indices(selected, selected >= 0)
    torch.testing.assert_close(ordered, torch.arange(2048, device=q.device).int().expand(4, -1))
    assert (counts == 2048).all()
    assert (stats["survivors"] == len(k) - seed_size).all()
    assert (stats["refinement_tiles"] == (len(k) - seed_size + 127) // 128).all()


@gpu
@pytest.mark.parametrize("capacity", [2049, 8193, 32769])
def test_radix_cutoff_and_original_id_ties(capacity: int) -> None:
    from relax.models.keye_vl2.dosa_parallel import select_codes

    torch.manual_seed(781)
    ids = torch.randperm(capacity, device="cuda").expand(3, -1)
    ordered_scores = torch.randint(1, 65536, (3, capacity), device="cuda", dtype=torch.int64)
    ordered_scores[1].fill_(32768)  # Massive boundary tie, shuffled token IDs.
    codes = (ordered_scores << 32) | (0xFFFFFFFF - ids)
    codes[2, 1000:] = 0  # Fewer than K valid entries, with zero padding.
    actual = select_codes(codes).sort().values
    expected = torch.topk(codes, 2048).values.sort().values
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@gpu
@pytest.mark.parametrize(
    "kind", ["best_anchor", "opposing_residuals", "single_anchor_residual", "collinear", "zero_anchors"]
)
def test_paper_plan_tightens_structured_residuals(kind: str) -> None:
    """Cover multi-anchor fits, degenerate anchors, and signed cancellation."""
    from relax.models.keye_vl2.dosa_kernels import prepare

    q, k, w, offsets, valid = _inputs(8193)
    q = torch.zeros_like(q).contiguous()
    w = torch.zeros_like(w).contiguous()
    for head, magnitude in enumerate((4, 3, 2, 1)):
        q[:, head, head] = magnitude
        w[:, head] = 1
    if kind == "best_anchor":
        q[:, 4, 1] = 1
        w[:, 4] = 1
    elif kind == "opposing_residuals":
        q[:, 4, 4], q[:, 5, 4] = 1, -1
        w[:, 4:6] = 1
    elif kind == "single_anchor_residual":
        for head, magnitude in enumerate((8, 7, 6, 5)):
            q[:, head, head] = magnitude
        q[:, 4, :2] = 1
        w[:, 4] = 1
    elif kind == "collinear":
        q.zero_()
        q[:, :5, 0] = torch.tensor([4, 3, 2, 1, 0.5], device=q.device)
        w[:, :5] = 1
    else:
        q.zero_()
        w[:, :5] = 1
    offsets[:] = len(k) - 1
    valid[:] = True
    seeds = (torch.arange(4096, device=q.device)[None].expand(4, -1) * 2).int().contiguous()
    visible = torch.full((4,), len(k), device=q.device, dtype=torch.int32)
    plan = torch.empty((4, 4 * 66 + 193), device=q.device)
    diagnostic = torch.empty((4, 16, 20), device=q.device)
    prepare[(4,)](
        q,
        w,
        k,
        seeds,
        visible,
        plan,
        q.stride(0),
        w.stride(0),
        4096,
        4,
        DIAG=diagnostic,
        DUMP=True,
        num_warps=4,
        enable_fp_fusion=False,
    )
    coefficients = diagnostic[:, :, :16].double()
    assert torch.isfinite(coefficients).all() and (coefficients >= 0).all()
    ranks = diagnostic[:, :, 19].long()
    for row in range(4):
        retained = q[row, ranks[row].argsort()[:4]].double()
        projected = coefficients[row, :, :4] @ retained
        omitted = (w[row] > 0) & (ranks[row] >= 4)
        exact_norm = ((q[row].double() - projected) * w[row, :, None].double()).norm(dim=-1)
        assert (diagnostic[row, omitted, 17].double() >= exact_norm[omitted]).all()
    if kind == "single_anchor_residual":
        # Paper projection uses one retained anchor, not the experimental NNLS fit.
        assert ((coefficients[:, 4, :4] > 0).sum(-1) == 1).all()
        assert (diagnostic[:, 4, 17] >= 1).all()
        assert (diagnostic[:, 4, 16] >= 1).all()
    allowance = plan[:, -1]
    if kind == "single_anchor_residual":
        assert ((allowance >= 1) & (allowance < 1.5)).all()
    elif kind != "opposing_residuals":
        # These omitted vectors have exact nonnegative retained-head fits.
        assert (allowance < 0.5).all()
    else:
        # Opposing unit residuals need coefficient 1 rather than 2, plus guards.
        assert ((allowance >= 1) & (allowance < 1.5)).all()
    _assert_selection(q, k, w, offsets, valid, heads=4)


@gpu
def test_diagnostic_head_swaps_preserve_certificate_and_ids(monkeypatch) -> None:
    """Exercise nonconsecutive retained IDs, including negative-weight
    heads."""
    from relax.models.keye_vl2 import dosa_kernels, dosa_parallel
    from relax.models.keye_vl2.indexer_scores import dense_indexer_scores

    q, k, w, offsets, valid = _inputs(8193)
    offsets[:] = len(k) - 1
    valid[:] = True
    order = torch.tensor([7, 1, 12, 3, 0, 2, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15], device=q.device)
    ranks = order.argsort().int().expand(4, -1).contiguous()
    w[:, 7] = -w[:, 7].abs() - 0.125
    upper = torch.empty((4, len(k) - 4096), device=q.device)
    original_prepare, original_screen = dosa_kernels.prepare, dosa_parallel.screen

    class Prepare:
        def __getitem__(self, grid):
            def call(*args, **kwargs):
                return original_prepare[grid](*args, HEAD_RANKS=ranks, OVERRIDE=True, **kwargs)

            return call

    class Screen:
        def __getitem__(self, grid):
            def call(*args, **kwargs):
                return original_screen[grid](*args, BOUNDS=upper, DUMP=True, **kwargs)

            return call

    monkeypatch.setattr(dosa_kernels, "prepare", Prepare())
    monkeypatch.setattr(dosa_parallel, "screen", Screen())
    _assert_selection(q, k, w, offsets, valid, heads=4)
    candidate = torch.ones(len(k), device=q.device, dtype=torch.bool)
    candidate[torch.arange(4096, device=q.device) * len(k) // 4096] = False
    assert torch.isfinite(upper).all()
    assert (upper >= dense_indexer_scores(q, k, w)[:, candidate].float()).all()


@gpu
@pytest.mark.parametrize("kind", ["signed", "ties", "zero", "poison_padding"])
def test_packed_selection_counts_and_original_id_boundary(kind: str) -> None:
    from relax.models.keye_vl2.dosa_parallel import select_records

    torch.manual_seed(187)
    width = 8193
    lengths = torch.tensor([8193, 4097, 13], device="cuda", dtype=torch.int32)
    ids = torch.randperm(width, device="cuda").expand(3, -1).clone()
    score = torch.randn(3, width, device="cuda", dtype=torch.bfloat16)
    if kind == "ties":
        score[:] = 1
    elif kind == "zero":
        score[:, ::2] = 0.0
        score[:, 1::2] = -0.0
    bits = score.view(torch.int16).to(torch.int32) & 0xFFFF
    bits = torch.where((bits & 0x7FFF) == 0, 0, bits)
    order = torch.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits ^ 0x8000)
    words = ((order >> 8) << 20 | ids).to(torch.uint32)
    low = (order & 255).to(torch.uint8)
    if kind == "poison_padding":
        # These records would win if any histogram/output loop ignored lengths.
        words[1, 4097:] = 0x0FF00000
        low[1, 4097:] = 255
        words[2, 13:] = 0x0FF00000
        low[2, 13:] = 255
    boundary = torch.empty_like(words)
    output = torch.empty((3, 2048), device="cuda", dtype=torch.int32)
    select_records[(3,)](words, low, lengths, boundary, output, width, 256, num_warps=4, enable_fp_fusion=False)
    for row, count in enumerate((8193, 4097, 13)):
        expected_codes = (order[row, :count].long() << 32) | (0xFFFFFFFF - ids[row, :count])
        winners = expected_codes.topk(min(2048, count)).indices
        expected = ids[row, :count][winners].int()
        torch.testing.assert_close(output[row, : len(expected)].sort().values, expected.sort().values)
        assert (output[row, len(expected) :] == -1).all()


@gpu
@pytest.mark.parametrize("width", [65536, 131072, 262144])
def test_long_prefix_complement_covers_nonseed_ids(width: int) -> None:
    from relax.models.keye_vl2.dosa_parallel import complement

    @triton.jit
    def inverse(OUT, BASE: tl.constexpr, CAP: tl.constexpr):
        t = tl.program_id(0) * 128 + tl.arange(0, 128)
        tl.store(OUT + t, complement(t, BASE, 4096), t < CAP)

    base = width - 127
    capacity = width - 4096
    actual = torch.empty(capacity, device="cuda", dtype=torch.int64)
    inverse[(triton.cdiv(capacity, 128),)](actual, base, capacity)
    seeds = torch.arange(4096, device="cuda", dtype=torch.int64) * base // 4096
    expected = torch.arange(width, device="cuda", dtype=torch.int64)
    expected = expected[~torch.isin(expected, seeds)]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@gpu
def test_packed_selection_cuda_graph_reuses_changing_lengths() -> None:
    from relax.models.keye_vl2.dosa_parallel import select_records

    width = 4097
    ids = torch.arange(width - 1, -1, -1, device="cuda", dtype=torch.int32)
    # All scores tie; shortest prefix must replace previous outputs and padding.
    words = ((0xBF << 20) | ids).to(torch.uint32)[None].contiguous()
    low = torch.zeros((1, width), device="cuda", dtype=torch.uint8)
    lengths = torch.tensor([width], device="cuda", dtype=torch.int32)
    boundary = torch.empty_like(words)
    output = torch.empty((1, 2048), device="cuda", dtype=torch.int32)

    def select() -> None:
        select_records[(1,)](words, low, lengths, boundary, output, width, 256, num_warps=4)

    select()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select()
    for count in (width, 13, 2048, width, 0, width):
        lengths.fill_(count)
        graph.replay()
        expected = ids[:count].sort().values[:2048]
        torch.testing.assert_close(output[0, : len(expected)].sort().values, expected)
        assert (output[0, len(expected) :] == -1).all()


@gpu
def test_fused_selector_graph_resets_compact_counters() -> None:
    from relax.models.keye_vl2.indexer_scores import dense_indexer_scores
    from relax.models.keye_vl2.sparse_attention import compact_indices
    from relax.models.keye_vl2.topk import dsa_topk_indices

    q, k, w, offsets, valid = _inputs(8193, rows=1)
    offsets[:] = len(k) - 1
    valid[:] = True
    original_weights = w.clone()
    _uniform_select(q, k, w, offsets, valid)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _uniform_select(q, k, w, offsets, valid)
    for end, zero_weights in ((8192, False), (13, False), (8192, True), (-1, True), (8192, False)):
        offsets.fill_(end)
        w.copy_(original_weights)
        if zero_weights:
            w.zero_()
        graph.replay()
        causal = torch.arange(len(k), device=q.device)[None] <= offsets[:, None]
        scores = dense_indexer_scores(q, k, w).masked_fill(~causal, torch.finfo(q.dtype).min)
        expected = dsa_topk_indices(scores)
        actual_compact = compact_indices(selected.clamp_min(0), selected >= 0)
        expected_compact = compact_indices(expected, causal.gather(1, expected.long()))
        for a, b in zip(actual_compact, expected_compact):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


@gpu
@pytest.mark.parametrize("seed_size", [4096, 12288])
def test_informative_seed_votes_original_ids_and_exact_selection(seed_size: int) -> None:
    from relax.models.keye_vl2.dosa import dosa_select
    from relax.models.keye_vl2.indexer_scores import dense_indexer_scores
    from relax.models.keye_vl2.topk import dsa_topk_indices

    q, k, w, offsets, valid = _inputs(16387)
    q[:] = q[0].clone()
    w[:] = w[0].clone()
    offsets[:] = len(k) - 1
    stats = {}
    actual = dosa_select(q, k, w, offsets, valid, heads=12, seed_size=seed_size, stats=stats)
    dense = dense_indexer_scores(q, k, w).masked_fill(~valid[None], -float("inf"))
    expected = dsa_topk_indices(dense)
    torch.testing.assert_close(actual.sort().values, expected.int().sort().values)
    votes = torch.zeros(len(k), device=q.device, dtype=torch.int32)
    votes.scatter_add_(0, expected[0].long(), torch.full((2048,), 4, device=q.device, dtype=torch.int32))
    torch.testing.assert_close(stats["seed_votes"][:-1], votes)
    ranked = votes.masked_fill(~valid, -1).argsort(descending=True, stable=True)[:seed_size]
    torch.testing.assert_close(stats["seed_ids"][0].sort().values, ranked.int().sort().values)
    for row in range(4):
        assert stats["seed_ids"][row].unique().numel() == seed_size
        candidates = stats["seed_order"][row // 4, seed_size:]
        assert not torch.isin(candidates[candidates < len(k)], stats["seed_ids"][row]).any()
        assert stats["seed_threshold"][row] == dense[row].topk(2048).values[-1]


@gpu
@pytest.mark.parametrize("seed_size", [4096, 12288])
def test_informative_seed_mixed_short_rows_and_padding(seed_size: int) -> None:
    from relax.models.keye_vl2.dosa import dosa_select
    from relax.models.keye_vl2.indexer_scores import dense_indexer_scores
    from relax.models.keye_vl2.sparse_attention import compact_indices
    from relax.models.keye_vl2.topk import dsa_topk_indices

    q, k, w, _, valid = _inputs(16387, rows=4)
    offsets = torch.tensor([-1, 2047, 4095, 16386], device=q.device)
    actual = dosa_select(q, k, w, offsets, valid, heads=8, seed_size=seed_size)
    causal = (torch.arange(len(k), device=q.device)[None] <= offsets[:, None]) & valid[None]
    dense = dense_indexer_scores(q, k, w).masked_fill(~causal, torch.finfo(q.dtype).min)
    expected = dsa_topk_indices(dense)
    for a, b in zip(
        compact_indices(actual.clamp_min(0), actual >= 0), compact_indices(expected, causal.gather(1, expected.long()))
    ):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@gpu
def test_informative_seed_carry_eligibility_and_reset() -> None:
    from relax.models.keye_vl2.dosa import DoSASeedState, dosa_select
    from relax.models.keye_vl2.indexer_scores import dense_indexer_scores
    from relax.models.keye_vl2.topk import dsa_topk_indices

    q, k, w, offsets, valid = _inputs(16387)
    offsets[:] = len(k) - 2
    causal = (torch.arange(len(k), device=q.device)[None] <= offsets[:, None]) & valid[None]
    previous = dsa_topk_indices(dense_indexer_scores(q, k, w).masked_fill(~causal, -float("inf")))
    state = DoSASeedState()
    state.publish(previous, offsets, len(k), causal.gather(1, previous.long()))
    original = state.selected.clone()
    offsets[:] = len(k) - 1
    stats = {}
    actual = dosa_select(q, k, w, offsets, valid, seed_state=state, stats=stats)
    assert stats["seed_source"] == 1
    votes = torch.zeros(len(k) + 1, device=q.device, dtype=torch.int32)
    votes.scatter_add_(0, original.long().flatten(), torch.ones(original.numel(), device=q.device, dtype=torch.int32))
    torch.testing.assert_close(stats["seed_votes"], votes)
    expected = dsa_topk_indices(dense_indexer_scores(q, k, w).masked_fill(~valid[None], -float("inf")))
    torch.testing.assert_close(actual.sort().values, expected.int().sort().values)
    # A prior ending after the common prefix must use cold representatives.
    offsets[:] = len(k) - 3
    dosa_select(q, k, w, offsets, valid, seed_state=state, stats=stats)
    assert stats["seed_source"] == 0
    state.clear()
    assert state.selected is None and state.visible_end is None
    # Native-size policy falls back for 8K; the caller owns original DSA.
    assert dosa_select(q, k[:8192], w, offsets, valid[:8192]) is None


@gpu
@pytest.mark.parametrize("kind", ["zero", "nan_query", "inf_key", "inf_weight"])
def test_informative_seed_original_scorer_exception_policy(kind: str) -> None:
    from relax.models.keye_vl2.dosa import dosa_select
    from relax.models.keye_vl2.indexer_scores import dense_indexer_scores
    from relax.models.keye_vl2.topk import dsa_topk_indices

    q, k, w, offsets, _ = _inputs(16387)
    offsets[:] = len(k) - 1
    if kind == "zero":
        w.zero_()
    elif kind == "nan_query":
        q[0, 0, 0] = float("nan")
    elif kind == "inf_key":
        k[4317, 0, 0] = float("inf")
    else:
        w[0, 0] = float("inf")
    actual = dosa_select(q, k, w, offsets, heads=12)
    expected = dsa_topk_indices(dense_indexer_scores(q, k, w))
    torch.testing.assert_close(actual.sort().values, expected.int().sort().values)


@gpu
def test_informative_selector_graph_recomputes_seed_and_visibility() -> None:
    from relax.models.keye_vl2.dosa import dosa_select
    from relax.models.keye_vl2.indexer_scores import dense_indexer_scores
    from relax.models.keye_vl2.sparse_attention import compact_indices
    from relax.models.keye_vl2.topk import dsa_topk_indices

    q, k, w, offsets, valid = _inputs(16387)
    offsets[:] = len(k) - 1
    original_weights = w.clone()
    dosa_select(q, k, w, offsets, valid)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = dosa_select(q, k, w, offsets, valid)
    for mixed, zero_weights in ((False, False), (True, False), (False, True), (False, False)):
        offsets.copy_(torch.tensor([-1, 2047, 4095, 16386] if mixed else [16386] * 4, device=q.device))
        w.copy_(original_weights)
        if zero_weights:
            w.zero_()
        graph.replay()
        causal = (torch.arange(len(k), device=q.device)[None] <= offsets[:, None]) & valid[None]
        dense = dense_indexer_scores(q, k, w).masked_fill(~causal, torch.finfo(q.dtype).min)
        expected = dsa_topk_indices(dense)
        for a, b in zip(
            compact_indices(actual.clamp_min(0), actual >= 0),
            compact_indices(expected, causal.gather(1, expected.long())),
        ):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
