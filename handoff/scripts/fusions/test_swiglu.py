import importlib.util, sys, time, torch
import torch.nn.functional as F
_s = importlib.util.spec_from_file_location("cute_swiglu", "/home/johnson/xllm/xllm/modules/fused_ops/cute/swiglu.py")
sg = importlib.util.module_from_spec(_s); _s.loader.exec_module(sg)
torch.manual_seed(0)
def timeit(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
for (R, E) in [(32768, 4096), (16384, 8192), (8192, 9216), (100, 64)]:
    h1 = (torch.randn(R, E, device="cuda") * 2).to(torch.bfloat16); h3 = torch.randn_like(h1); g = torch.randn_like(h1)
    def ref_fwd():
        hid = F.silu(h1); return torch.mul(hid, h3, out=hid)
    def ref_bwd():
        h1s = F.silu(h1); h = h1s * h3
        h1s_grad = g * h3; h3_grad = g * h1s
        return h, torch.ops.aten.silu_backward(h1s_grad, h1), h3_grad
    o = ref_fwd(); o2 = sg.swiglu_fwd(h1, h3)
    h, d1, d3 = ref_bwd(); h2, d12, d32 = sg.swiglu_bwd(h1, h3, g)
    print(f"R{R} E{E}: fwd err {(o2.float()-o.float()).abs().max().item():.2e} (eq {(o2==o).float().mean().item():.4f}) | bwd h err {(h2.float()-h.float()).abs().max().item():.2e} d1 err {(d12.float()-d1.float()).abs().max().item():.2e} (eq {(d12==d1).float().mean().item():.4f}) d3 err {(d32.float()-d3.float()).abs().max().item():.2e} (eq {(d32==d3).float().mean().item():.4f})")
    print(f"   fwd torch {timeit(ref_fwd):.0f} cute {timeit(lambda: sg.swiglu_fwd(h1, h3)):.0f} us | bwd torch {timeit(ref_bwd):.0f} cute {timeit(lambda: sg.swiglu_bwd(h1, h3, g)):.0f} us")
