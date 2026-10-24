import importlib.util, sys, time, types, math, torch
import torch.nn.functional as F
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
sys.path.insert(0, "/home/johnson/xllm"); sys.path.insert(0, "/home/johnson/scratch/xext")
_s = importlib.util.spec_from_file_location("cute_gate", "/home/johnson/xllm/xllm/modules/fused_ops/cute/attn_gate.py")
ag = importlib.util.module_from_spec(_s); _s.loader.exec_module(ag)
from xllm.models.fused_blocks.utils import rmsnorm_fwd, mem_effn_rmsnorm_bwd
torch.manual_seed(0)
def timeit(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
eps = 1e-6
for (B, L, H, V, gate) in [(2, 16384, 8, 256, "softplus"), (1, 8192, 32, 128, "softplus"), (1, 16384, 4, 2048, "softplus"), (1, 1024, 4, 512, "silu"), (1, 100, 3, 256, "softplus")]:
    sca = torch.randn(B, L, H * V, device="cuda").to(torch.bfloat16); awk = torch.randn_like(sca); rmx = (torch.randn_like(sca) * 3)
    g = torch.randn_like(sca)
    fn = {"silu": F.silu, "softplus": lambda t: F.softplus(t, beta=math.log(2))}[gate]
    fn_bwd = {"silu": torch.ops.aten.silu_backward, "softplus": lambda a, b: torch.ops.aten.softplus_backward(a, b, beta=math.log(2), threshold=20)}[gate]
    def ref_fwd():
        r = fn(rmx)
        attn, _ = rmsnorm_fwd(sca + awk, None, H, eps)
        return torch.mul(attn, r)
    def ref_bwd():
        r = fn(rmx)
        attn_norm, rstd = rmsnorm_fwd(sca + awk, None, H, eps)
        attn_norm_grad = torch.mul(g, r)
        ds = mem_effn_rmsnorm_bwd(attn_norm_grad, attn_norm, rstd, H)
        r_grad = torch.mul(g, attn_norm)
        return ds, fn_bwd(r_grad, rmx)
    out = ref_fwd(); out2 = ag.attn_gate_fwd(sca, awk, rmx, H, gate, eps)
    ds, dr = ref_bwd(); ds2, dr2, _ = ag.attn_gate_bwd(g, sca, awk, rmx, H, gate, eps, want_attn=True)
    # fp32 autograd reference
    sf, af, rf = [t.float().requires_grad_(True) for t in (sca, awk, rmx)]
    s = (sf + af).view(B, L, H, V); n = s * torch.rsqrt((s * s).mean(-1, keepdim=True) + eps)
    o = (n.view(B, L, H * V) * fn(rf)); (o * g.float()).sum().backward()
    print(f"B{B} L{L} H{H} V{V} {gate}: fwd err vs torch {(out2.float()-out.float()).abs().max().item():.2e} vs fp32 {(out2.float()-o).abs().max().item():.2e} (torch vs fp32 {(out.float()-o).abs().max().item():.2e}, max {o.abs().max().item():.1f})")
    print(f"   bwd ds err vs torch {(ds2.float()-ds.float()).abs().max().item():.2e} vs fp32 {(ds2.float()-sf.grad).abs().max().item():.2e} (torch vs fp32 {(ds.float()-sf.grad).abs().max().item():.2e}, max {sf.grad.abs().max().item():.2f}); "
          f"dr err vs torch {(dr2.float()-dr.float()).abs().max().item():.2e} vs fp32 {(dr2.float()-rf.grad).abs().max().item():.2e} (torch vs fp32 {(dr.float()-rf.grad).abs().max().item():.2e}, max {rf.grad.abs().max().item():.2f})")
    print(f"   time fwd torch {timeit(ref_fwd):.0f} cute {timeit(lambda: ag.attn_gate_fwd(sca, awk, rmx, H, gate, eps)):.0f} us | bwd torch {timeit(ref_bwd):.0f} cute {timeit(lambda: ag.attn_gate_bwd(g, sca, awk, rmx, H, gate, eps, want_attn=True)):.0f} us")
