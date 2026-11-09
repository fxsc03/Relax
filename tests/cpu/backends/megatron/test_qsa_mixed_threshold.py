# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise production dispatch with external GPU kernels replaced, not kernel
numerics."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def mixed_dispatcher(monkeypatch):
    """Import the dispatcher on CPU-only hosts without importing Triton
    kernels."""
    from relax.utils.env import Envs

    monkeypatch.delenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", raising=False)
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_LONG_BACKEND", "grouped")
    prefix = "relax.models.qwen3_8.ops.qsa_mixed_ops"
    flex = ModuleType(f"{prefix}.flex")
    grouped = ModuleType(f"{prefix}.grouped")

    def no_kernel(*_args, **_kwargs):
        pytest.fail("A dispatch-only test must replace GPU execution")

    flex.attention = flex.attention_auto = grouped.attention_bf16 = no_kernel
    path = Path(__file__).resolve().parents[4] / "relax/models/qwen3_8/ops/qsa_mixed_ops/mixed.py"
    spec = importlib.util.spec_from_file_location("qsa_mixed_dispatch_under_test", path)
    mixed = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, flex.__name__, flex)
        imports.setitem(sys.modules, grouped.__name__, grouped)
        geometry_spec = importlib.util.spec_from_file_location(f"{prefix}.geometry", path.with_name("geometry.py"))
        geometry = importlib.util.module_from_spec(geometry_spec)
        imports.setitem(sys.modules, geometry_spec.name, geometry)
        geometry_spec.loader.exec_module(geometry)
        spec.loader.exec_module(mixed)
    assert mixed.Envs is Envs
    return mixed


@pytest.mark.parametrize(
    "q_heads,kv_heads,q_dim,k_dim,valid",
    [
        (12, 1, 256, 256, True),
        (24, 2, 256, 256, True),
        (6, 1, 256, 256, True),
        (3, 1, 256, 256, True),
        (8, 2, 256, 256, True),
        (0, 1, 256, 256, False),
        (12, 0, 256, 256, False),
        (3, 2, 256, 256, False),
        (12, 1, 128, 256, False),
        (12, 1, 256, 128, False),
    ],
)
def test_grouped_long_geometry_contract(monkeypatch, mixed_dispatcher, q_heads, kv_heads, q_dim, k_dim, valid):
    q = torch.empty((7, q_heads, q_dim))
    k = torch.empty((7, kv_heads, k_dim))
    indices = torch.empty((7, 2052), dtype=torch.int32)
    if valid:
        monkeypatch.setattr(mixed_dispatcher, "attention_bf16", lambda q, *args, **kwargs: q)
        assert mixed_dispatcher.long_attention(q, k, k, indices, 0.0625) is q
    else:
        # The fixture's kernel sentinel fails if validation accidentally admits the input.
        with pytest.raises(NotImplementedError, match="backend=grouped"):
            mixed_dispatcher.long_attention(q, k, k, indices, 0.0625)


@pytest.mark.parametrize(
    "limit,length,expected",
    [
        (None, 2052, "auto"),
        (None, 16384, "auto"),
        (None, 16385, "grouped"),
        ("0", 2051, "auto"),
        ("0", 2052, "grouped"),
        ("4096", 4096, "auto"),
        ("4096", 4097, "grouped"),
    ],
)
def test_mixed_threshold_keeps_dense_short_and_routes_sparse(monkeypatch, mixed_dispatcher, limit, length, expected):
    mixed = mixed_dispatcher
    if limit is None:
        monkeypatch.delenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", raising=False)
    else:
        monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", limit)
    seen = []

    def call(kind):
        def forward(q, *_args, **_kwargs):
            seen.append(kind)
            return q

        return forward

    monkeypatch.setattr(mixed, "attention_auto", call("auto"))
    monkeypatch.setattr(mixed, "long_attention", call("grouped"))
    q = torch.empty(length, 12, 256, device="meta")
    kv = torch.empty(length, 1, 256, device="meta")
    indices = torch.empty(length, 2052, dtype=torch.int32, device="meta")
    assert mixed.attention(q, kv, kv, indices, 256**-0.5) is q
    assert seen == [expected]


@pytest.mark.parametrize("limit", [-1, 16385])
def test_mixed_threshold_rejects_unsupported_flex_layout(monkeypatch, mixed_dispatcher, limit):
    mixed = mixed_dispatcher
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", str(limit))
    q = torch.empty(2052, 12, 256, device="meta")
    kv = torch.empty(2052, 1, 256, device="meta")
    indices = torch.empty(2052, 2052, dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="maximum length"):
        mixed.attention(q, kv, kv, indices, 256**-0.5)


@pytest.mark.parametrize("limit", [2047, 2051])
@pytest.mark.parametrize("length", [2047, 2048, 2051, 2052])
def test_explicit_flash_boundary_controls_legacy_long_branch(monkeypatch, mixed_dispatcher, limit, length):
    mixed = mixed_dispatcher
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", str(limit))
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", "0")
    seen = []

    def route(name):
        def call(q, *_args, **_kwargs):
            seen.append(name)
            return q

        return call

    monkeypatch.setattr(mixed, "attention_auto", route("flash"))
    monkeypatch.setattr(mixed, "long_attention", route("grouped"))
    q = torch.empty(length, 12, 256, device="meta")
    kv = torch.empty(length, 1, 256, device="meta")
    ids = torch.empty(length, 2052, dtype=torch.int32, device="meta")
    assert mixed.attention(q, kv, kv, ids, 256**-0.5) is q
    assert seen == ["flash" if length <= limit else "grouped"]


@pytest.mark.parametrize("flash,flex", [(2052, 0), (-2, 0), (2051, 16384)])
def test_explicit_flash_boundary_rejects_incompatible_settings(monkeypatch, mixed_dispatcher, flash, flex):
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", str(flash))
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", str(flex))
    q = torch.empty(2052, 12, 256, device="meta")
    kv = torch.empty(2052, 1, 256, device="meta")
    ids = torch.empty(2052, 2052, dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="dense selector budget|Flex maximum length zero"):
        mixed_dispatcher.attention(q, kv, kv, ids, 256**-0.5)


@pytest.mark.parametrize("length,expected", [(35, "flash"), (36, "flex"), (65, "grouped")])
def test_custom_budget_plan_reaches_single_branch_executor(monkeypatch, mixed_dispatcher, length, expected):
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", "64")
    seen = []

    def auto(q, *args, plan):
        seen.append("flash" if plan.flash else "flex")
        return q

    monkeypatch.setattr(mixed_dispatcher, "attention_auto", auto)
    monkeypatch.setattr(mixed_dispatcher, "long_attention", lambda q, *args: (seen.append("grouped"), q)[1])
    q, kv = torch.empty(length, 12, 256, device="meta"), torch.empty(length, 1, 256, device="meta")
    ids = torch.empty(length, 36, dtype=torch.int32, device="meta")
    assert mixed_dispatcher.attention(q, kv, kv, ids, 0.0625) is q
    assert seen == [expected]


def test_packed_all_branches_preserve_offsets_order_and_qkv_gradients(monkeypatch, mixed_dispatcher):
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", "20")
    bounds = (0, 25, 25, 32, 47, 50, 71, 71, 89)
    total = bounds[-1]
    q, k, v = [torch.randn(total, heads, 4, requires_grad=True) for heads in (2, 1, 1)]
    ids = torch.full((total, 12), -1, dtype=torch.int32)
    for start, end in zip(bounds, bounds[1:]):
        ids[start:end, 0] = start
    ids._qsa_bounds = bounds
    calls = []

    def flash(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
        assert cu_q.tolist() == cu_k.tolist() == [0, 7, 10]
        assert max_q == max_k == 7 and kwargs["causal"]
        calls.append("flash")
        return q + 2 * k + 3 * v

    def sparse(kind, factor, sizes, origins):
        def call(q, k, v, selected, scale):
            calls.append(kind)
            assert selected._qsa_bounds == sizes
            offset = 0
            for length, origin in origins:
                assert torch.equal(selected[offset : offset + length, 0], torch.full((length,), origin))
                assert (selected[offset : offset + length, 1:] == -1).all()
                offset += length
            if kind == "flex":
                assert q.shape[0] == 16384
                assert (selected[offset:] == -1).all()
                assert selected._qsa_layout[-1] == 16384
            return factor * (q + 2 * k + 3 * v)

        return call

    monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace(flash_attn_varlen_func=flash))
    monkeypatch.setitem(
        sys.modules,
        "relax.models.qwen3_8.ops.qsa_mixed_ops.flex",
        SimpleNamespace(layout=lambda bounds, device: (None, None, None, max(bounds))),
    )
    monkeypatch.setattr(mixed_dispatcher, "flex", sparse("flex", 2, (0, 15, 33, 16384), [(15, 0), (18, 15)]))
    monkeypatch.setattr(mixed_dispatcher, "long_attention", sparse("grouped", 3, (0, 25, 46), [(25, 0), (21, 25)]))
    output = mixed_dispatcher.attention(q, k, v, ids, 0.5)
    factor = torch.tensor([3] * 25 + [1] * 7 + [2] * 15 + [1] * 3 + [3] * 21 + [2] * 18).view(-1, 1, 1)
    torch.testing.assert_close(output, factor * (q + 2 * k + 3 * v))
    upstream = torch.randn_like(q)
    output.backward(upstream)
    torch.testing.assert_close(q.grad, upstream * factor)
    torch.testing.assert_close(k.grad, (upstream * factor).sum(dim=1, keepdim=True) * 2)
    torch.testing.assert_close(v.grad, (upstream * factor).sum(dim=1, keepdim=True) * 3)
    assert calls == ["flash", "flex", "grouped"]


def test_empty_input_keeps_early_return_before_invalid_configuration(monkeypatch, mixed_dispatcher):
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", "invalid")
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", "invalid")
    q = torch.empty(0, 12, 256, requires_grad=True)
    output = mixed_dispatcher.attention(q, q, q, torch.empty(0, 36), 0.0625)
    assert output is not q
    output.sum().backward()
    assert q.grad is not None


def test_flash_configuration_is_validated_before_reading_flex(monkeypatch, mixed_dispatcher):
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", "36")
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", "invalid")
    q = torch.empty(1, 12, 256, device="meta")
    with pytest.raises(ValueError, match="dense selector budget"):
        mixed_dispatcher.attention(q, q, q, torch.empty(1, 36, device="meta"), 0.0625)
