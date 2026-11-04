# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye BF16 DoSA requires CUDA")


def test_backend_config_and_actual_freeze() -> None:
    from relax.models.keye_vl2.dosa import backend_config, dosa_select, frozen_indexer_enabled

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
    assert dosa_select(torch.empty(1, 16, 64), torch.empty(1, 1, 64), torch.empty(1, 16), torch.zeros(1)) is None


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
    from relax.models.keye_vl2.dosa import dosa_select

    q, k, w, offsets, valid = _inputs(4097)
    assert dosa_select(q.float(), k.float(), w.float(), offsets) is None
    assert dosa_select(q[:, :8], k, w[:, :8], offsets) is None
    assert dosa_select(q, k, w, offsets.cpu()) is None
    assert dosa_select(q, k, w, offsets, valid.cpu()) is None


def _assert_selection(q, k, w, offsets, valid, heads=4):
    from relax.models.keye_vl2.dosa import dosa_select
    from relax.models.keye_vl2.indexer_scores import dense_indexer_scores
    from relax.models.keye_vl2.sparse_attention import compact_indices
    from relax.models.keye_vl2.topk import dsa_topk_indices

    stats = {}
    actual = dosa_select(q, k, w, offsets, valid, heads=heads, stats=stats)
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
    actual = attention(positions, hidden, batch)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
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
    from relax.models.keye_vl2.dosa import dosa_select
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
    dosa_select(q, k, w, offsets, valid, heads=heads)
    seeds = torch.arange(4096, device=q.device) * len(k) // 4096
    candidate = torch.ones(len(k), device=q.device, dtype=torch.bool)
    candidate[seeds] = False
    exact = dense_indexer_scores(q, k, w)[:, candidate].float()
    assert (upper >= exact).all()


@gpu
@pytest.mark.parametrize("seed_size", [2048, 4096, 8192])
def test_cross_tile_packing_full_capacity_and_seed_sizes(seed_size: int) -> None:
    from relax.models.keye_vl2.dosa import dosa_select
    from relax.models.keye_vl2.sparse_attention import compact_indices

    q, k, w, _, _ = _inputs(16387)
    q.zero_()  # All ties: every non-seed key must survive, never truncate.
    offsets = torch.full((4,), len(k) - 1, device=q.device)
    stats = {}
    selected = dosa_select(q, k, w, offsets, seed_size=seed_size, stats=stats)
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
