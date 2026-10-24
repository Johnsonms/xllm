import importlib.util, sys, time, types, torch
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
sys.path.insert(0, "/home/johnson/xllm"); sys.path.insert(0, "/home/johnson/scratch/xext")
_s = importlib.util.spec_from_file_location("cute_ln", "/home/johnson/xllm/xllm/modules/fused_ops/cute/layer_norm.py")
ln = importlib.util.module_from_spec(_s); _s.loader.exec_module(ln)
from xllm.models.fused_blocks.utils import layernorm_fwd, layernorm_bwd
torch.manual_seed(0)
def timeit(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
eps = 1e-5
for (B, L, D, G) in [(2, 16384, 1536, 2), (1, 8192, 2560, 2), (1, 16384, 4096, 4), (1, 100, 512, 2), (1, 64, 4096, 1)]:
    x = (torch.randn(B, L, D, device="cuda") * 2 + 1).to(torch.bfloat16)
    w = (torch.randn(D, device="cuda") * 0.1).to(torch.bfloat16); b = (torch.randn(D, device="cuda") * 0.1).to(torch.bfloat16)
    dy = torch.randn_like(x)
    y, m, r = layernorm_fwd(x, w, b, G, eps); y2, m2, r2 = ln.ln_fwd(x, w, b, G, eps)
    dx, dw, db = layernorm_bwd(dy, x, m, r, w, b, G); dx2, dw2, db2 = ln.ln_bwd(dy, x, m2, r2, w, G)
    xf = x.float().requires_grad_(True); wf = w.float().requires_grad_(True); bf = b.float().requires_grad_(True)
    xv = xf.view(B, L, G, D // G); mu = xv.mean(-1, keepdim=True); var = ((xv - mu) ** 2).mean(-1, keepdim=True)
    yr = ((xv - mu) * torch.rsqrt(var + eps)).view(B, L, D) * (wf + 1.0) + bf
    (yr * dy.float()).sum().backward()
    print(f"B{B} L{L} D{D} G{G}: y err vs ext {(y2.float()-y.float()).abs().max().item():.2e} vs fp32 {(y2.float()-yr).abs().max().item():.2e} (ext vs fp32 {(y.float()-yr).abs().max().item():.2e}); mean err {(m2-m.view(-1,G)).abs().max().item():.2e} rstd err {(r2-r.view(-1,G)).abs().max().item():.2e}")
    print(f"   dx err vs ext {(dx2.float()-dx.float()).abs().max().item():.2e} vs fp32 {(dx2.float()-xf.grad).abs().max().item():.2e} (ext vs fp32 {(dx.float()-xf.grad).abs().max().item():.2e}, max {xf.grad.abs().max().item():.2f}); dw err vs fp32 {(dw2.float()-wf.grad).abs().max().item():.2e} (ext {(dw.float()-wf.grad).abs().max().item():.2e}, max {wf.grad.abs().max().item():.1f}) db err {(db2.float()-bf.grad).abs().max().item():.2e} (ext {(db.float()-bf.grad).abs().max().item():.2e})")
    print(f"   fwd ext {timeit(lambda: layernorm_fwd(x, w, b, G, eps)):.0f} cute {timeit(lambda: ln.ln_fwd(x, w, b, G, eps)):.0f} us | bwd ext {timeit(lambda: layernorm_bwd(dy, x, m, r, w, b, G)):.0f} cute {timeit(lambda: ln.ln_bwd(dy, x, m2, r2, w, G)):.0f} us")
