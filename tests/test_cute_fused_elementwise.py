"""CuTe DSL fused kernels of the gekko block (any arch) against their torch / xllm_extension references:
rotary embedding, softmax backward with accumulation, SwiGLU, group RMS / layer norm forward."""

import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _ulp_close(a, b, ulps=1):
    """bf16 results within `ulps` units in the last place of the larger magnitude."""
    tol = ulps * torch.finfo(torch.bfloat16).eps * torch.maximum(a.float().abs(), b.float().abs()).clamp_min(1e-3)
    assert ((a.float() - b.float()).abs() <= tol + 1e-6).all(), (a.float() - b.float()).abs().max()


@pytest.mark.parametrize("B,L,H,D,R", [(2, 512, 8, 256, 128), (1, 1024, 4, 128, 128), (2, 256, 3, 64, 32)])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("backward", [False, True])
def test_rope(B, L, H, D, R, strided, backward):
    from xllm.models.fused_blocks import utils
    from xllm.modules.fused_ops.cute.rope import rope
    torch.manual_seed(0)
    phase = torch.randn(L, R // 2, device="cuda")
    f = torch.polar(torch.ones_like(phase), phase)
    base = torch.randn(B, L, H, D + (64 if strided else 0), device="cuda").bfloat16()
    x = base[..., :D]
    x0 = x.clone()
    ref = utils.apply_rotary_embedding(x[..., :R], f, backward)
    ref = torch.cat((ref, x[..., R:]), -1) if R < D else ref
    out = rope(x, f, R, backward)
    assert torch.equal(x, x0)
    _ulp_close(out, ref)


@pytest.mark.parametrize("shape", [(2, 512, 8, 256), (1, 256, 32, 128), (3, 128, 2, 64)])
@pytest.mark.parametrize("with_add", [False, True])
def test_softmax_bwd_acc(shape, with_add):
    from xllm.modules.fused_ops.cute.softmax_bwd import softmax_bwd_acc_
    torch.manual_seed(0)
    x = (torch.randn(*shape, device="cuda") * 3).bfloat16()
    g, acc = torch.randn_like(x), torch.randn_like(x)
    add = torch.randn_like(x) if with_add else None
    y = F.softmax(x.double(), -1)
    dx = y * (g.double() - (g.double() * y).sum(-1, keepdim=True))
    ref = acc.double() + dx + (add.double() if with_add else 0)
    out = softmax_bwd_acc_(acc.clone(), x, g, add)
    torch.testing.assert_close(out.double(), ref, atol=3e-2, rtol=1e-2)


def test_swiglu():
    from xllm.modules.fused_ops.cute.swiglu import swiglu_bwd, swiglu_fwd
    torch.manual_seed(0)
    h1 = (torch.randn(1024, 512, device="cuda") * 2).bfloat16()
    h3, g = torch.randn_like(h1), torch.randn_like(h1)
    a, c, gg = h1.double(), h3.double(), g.double()
    s = torch.sigmoid(a)
    torch.testing.assert_close(swiglu_fwd(h1, h3).double(), a * s * c, atol=2e-2, rtol=1e-2)
    d1, d3 = swiglu_bwd(g, h1, h3)
    torch.testing.assert_close(d1.double(), gg * c * s * (1 + a * (1 - s)), atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(d3.double(), gg * a * s, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("D,G,affine", [(1024, 4, False), (2048, 8, True), (4096, 32, True), (1024, 8, False)])
def test_group_rms_norm_fwd(D, G, affine):
    import xllm_extension.ops as ops
    from xllm.modules.fused_ops.cute.group_norm import group_norm_fwd
    torch.manual_seed(0)
    x = torch.randn(2, 512, D, device="cuda").bfloat16()
    w = torch.randn(D, device="cuda").bfloat16()
    ry, rr = ops.group_rms_norm_fwd_affine(x, D, G, w, 1e-6) if affine else ops.group_rms_norm_fwd(x, D, G, 1e-6)
    y, r = group_norm_fwd(x, G, 1e-6, w if affine else None)
    assert r.shape == rr.shape
    torch.testing.assert_close(r, rr, atol=0, rtol=1e-6)
    _ulp_close(y, ry)


def test_group_layer_norm_fwd():
    import xllm_extension.ops as ops
    from xllm.modules.fused_ops.cute.group_norm import group_norm_fwd
    torch.manual_seed(0)
    D, G = 1536, 2
    x = torch.randn(2, 512, D, device="cuda").bfloat16()
    w, b = torch.randn(D, device="cuda").bfloat16(), torch.randn(D, device="cuda").bfloat16()
    ry, rm, rr = ops.group_layer_norm_fwd_affine(x, D, G, w, b, 1e-6)
    y, m, r = group_norm_fwd(x, G, 1e-6, w, b, layer=True)
    assert m.shape == rm.shape and r.shape == rr.shape
    torch.testing.assert_close(m, rm, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(r, rr, atol=0, rtol=1e-6)
    _ulp_close(y, ry)
