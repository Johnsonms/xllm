import importlib.util, sys, time, types, torch
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
sys.path.insert(0, "/home/johnson/xllm"); sys.path.insert(0, "/home/johnson/scratch/xext")
_s = importlib.util.spec_from_file_location("cute_qk", "/home/johnson/xllm/xllm/modules/fused_ops/cute/qk_prep.py")
qk = importlib.util.module_from_spec(_s); _s.loader.exec_module(qk)
import torch.nn.functional as F
from xllm.models.fused_blocks.utils import rmsnorm_fwd, rmsnorm_bwd, mem_effn_rmsnorm_bwd, apply_rope
torch.manual_seed(0)
def freqs(L, R, base=500000.0):
    f = 1.0 / (base ** (torch.arange(0, R, 2, device="cuda").float() / R))
    return torch.polar(torch.ones(L, R // 2, device="cuda"), torch.outer(torch.arange(L, device="cuda").float(), f))
def timeit(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
eps = 1e-6
for (B, L, G, S, R, affine) in [(2, 16384, 8, 256, 128, True), (2, 16384, 4, 256, 128, False), (1, 8192, 32, 128, 128, True), (1, 16384, 4, 256, 256, True), (1, 128, 2, 64, 32, False)]:
    x = torch.randn(B, L, G * S, device="cuda").to(torch.bfloat16)
    w = (torch.randn(G * S, device="cuda") * 0.1).to(torch.bfloat16) if affine else None
    fc = freqs(L, R)
    # torch path
    def ref_fwd():
        y, rstd = rmsnorm_fwd(x, w, G, eps)
        y = y.view(B, L, G, S)
        yr = apply_rope(y, fc, S, R, False)
        p = F.softmax(x.view(B, L, G, S), dim=-1, dtype=torch.float32)
        return yr, y, p.to(x.dtype), rstd, p
    yr, y, p, rstd, p32 = ref_fwd()
    y2, pre2, p2, rstd2 = qk.qk_prep_fwd(x, G, w, fc, R, eps, want_pre=True, want_sm=True)
    print(f"B{B} L{L} G{G} S{S} R{R} aff={affine} fwd: y err {(y2.float()-yr.float()).abs().max().item():.2e} pre err {(pre2.float()-y.float()).abs().max().item():.2e} "
          f"p err {(p2.float()-p.float()).abs().max().item():.2e} rstd err {(rstd2-rstd.view(-1, G)).abs().max().item():.2e}")
    dy = torch.randn(B, L, G, S, device="cuda").to(torch.bfloat16); dp = torch.randn(B, L, G, S, device="cuda").to(torch.bfloat16)
    ex = torch.randn(B, L, G, S, device="cuda").to(torch.bfloat16)
    def ref_bwd():
        g1 = apply_rope(dy, fc, S, R, True).reshape(B, L, G * S)
        if affine:
            dx, dw = rmsnorm_bwd(g1, x, rstd, w, G)
        else:
            dx, dw = mem_effn_rmsnorm_bwd(g1, y.reshape(B, L, G * S), rstd, G), None
        dxs = torch.ops.aten._softmax_backward_data(dp.float(), p32, -1, torch.float32).to(x.dtype)
        dx = dx + dxs.reshape(B, L, G * S) + ex.reshape(B, L, G * S)
        return dx, dw
    dx, dw = ref_bwd()
    dx2, dw2 = qk.qk_prep_bwd(dy, dp, ex, x, G, w, fc, R, eps)
    # fp32 reference for the backward (autograd)
    xf = x.float().detach().requires_grad_(True); wf = (w.float() + 1.0) if affine else None
    xv = xf.view(B, L, G, S); yv = xv * torch.rsqrt((xv * xv).mean(-1, keepdim=True) + eps)
    if affine: yv = yv * wf.view(G, S)
    yrr = torch.view_as_real(torch.view_as_complex(yv[..., :R].reshape(B, L, G, R // 2, 2)) * fc.unsqueeze(1)).flatten(3)
    yrr = torch.cat((yrr, yv[..., R:]), -1)
    pv = F.softmax(xv, -1)
    (yrr * dy.float()).sum().backward(); (pv * dp.float()).sum().backward()
    dxr = xf.grad.view(B, L, G * S) + ex.float().view(B, L, G * S)
    print(f"   bwd: dx err vs torch {(dx2.float()-dx.float()).abs().max().item():.2e} vs fp32 {(dx2.float()-dxr).abs().max().item():.2e} (torch vs fp32 {(dx.float()-dxr).abs().max().item():.2e}, max {dxr.abs().max().item():.2f})"
          + (f" dw err vs torch {(dw2.float()-dw.float()).abs().max().item():.2e} (max {dw.float().abs().max().item():.1f})" if affine else ""))
    print(f"   time fwd torch {timeit(ref_fwd):.0f} us cute {timeit(lambda: qk.qk_prep_fwd(x, G, w, fc, R, eps, want_pre=True, want_sm=True)):.0f} us | bwd torch {timeit(ref_bwd):.0f} us cute {timeit(lambda: qk.qk_prep_bwd(dy, dp, ex, x, G, w, fc, R, eps)):.0f} us")
