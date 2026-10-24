"""swift SCA fwd + bwd at the gekko-7.1B layer shape (B1 L8192 H4 S256 V2048, C4096), GPU us per call."""
import sys, types, torch
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
from torch.profiler import profile, ProfilerActivity
from xllm.modules.fused_ops.attention.sca import sliding_chunk_attention_fwd, sliding_chunk_attention_bwd
B, L, H, KV, S, V, C = 1, 8192, 4, 4, 256, 2048, 4096
g = torch.Generator(device="cuda").manual_seed(0)
q = torch.randn(B, L, H, S, device="cuda", generator=g).bfloat16(); k = torch.randn(B, L, KV, S, device="cuda", generator=g).bfloat16()
v = torch.randn(B, L, KV, V, device="cuda", generator=g).bfloat16(); dy = torch.randn(B, L, H, V, device="cuda", generator=g).bfloat16()
seg = torch.cumsum((torch.rand(B, L, device="cuda", generator=g) < 1 / 3000).long(), 1)
def gt(fn, it=5):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(it): fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in p.key_averages()) / it
fwd = lambda: sliding_chunk_attention_fwd(q, k, v, C, S ** -0.5, None, None, None, seg, 0.0, 0, "swift", True)
y, yb, w = fwd()
bwd = lambda: sliding_chunk_attention_bwd(dy, q, k, v, yb, w, C, S ** -0.5, None, None, None, seg, 0, False, "swift")
gq, gk, gv = bwd()[:3]
torch.save({"y": y, "gq": gq, "gk": gk, "gv": gv}, sys.argv[1]) if len(sys.argv) > 1 else None
print(f"swift 7.1B layer: fwd {gt(fwd):.0f} us, bwd {gt(bwd):.0f} us; w shape {tuple(w.shape)}")
