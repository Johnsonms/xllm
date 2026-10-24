import importlib.util, sys, time, types, torch
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
sys.path.insert(0, "/home/johnson/xllm"); sys.path.insert(0, "/home/johnson/scratch/xext")
_s = importlib.util.spec_from_file_location("cute_rope", "/home/johnson/xllm/xllm/modules/fused_ops/cute/rope.py")
rp = importlib.util.module_from_spec(_s); _s.loader.exec_module(rp)
from xllm.modules.rotary_positional_embedding import apply_rotary_embedding
from xllm.models.fused_blocks.utils import apply_rope
torch.manual_seed(0)
def freqs(L, R, base=500000.0):
    f = 1.0 / (base ** (torch.arange(0, R, 2, device="cuda").float() / R))
    t = torch.arange(L, device="cuda").float()
    return torch.polar(torch.ones(L, R // 2, device="cuda"), torch.outer(t, f))
for (B, L, N, D, R) in [(2, 16384, 8, 256, 128), (2, 16384, 4, 256, 128), (1, 8192, 32, 128, 128), (1, 16384, 4, 256, 256), (1, 64, 2, 64, 16)]:
    x = torch.randn(B, L, N, D, device="cuda").to(torch.bfloat16)
    fc = freqs(L, R)
    for bwd in (False, True):
        ref = apply_rope(x, fc, D, R, bwd)
        out = rp.rope(x, fc, R, bwd)
        err = (out.float() - ref.float()).abs().max().item()
        print(f"B{B} L{L} N{N} D{D} R{R} bwd={bwd}: max err {err:.3e} (ref max {ref.abs().max().item():.2f}) exact-frac {(out == ref).float().mean().item():.4f}")
    def timeit(fn, it=20):
        for _ in range(3): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(it): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
    print(f"   torch {timeit(lambda: apply_rope(x, fc, D, R, False)):.0f} us, cute {timeit(lambda: rp.rope(x, fc, R, False)):.0f} us ({2 * x.numel() * 2 / 1e3 / timeit(lambda: rp.rope(x, fc, R, False)):.2f} TB/s)")
