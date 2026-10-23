import sys
import torch
import cutlass.cute as cute
from cutlass import BFloat16, Uint8, Int32, Float32
sys.path.insert(0, "/home/johnson/xllm/xllm/modules/fused_ops/cute")
from adaptive_working_memory import AwmKeyLse


def fake(dtype, align=16):
    return cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(),), assumed_align=align)


B, K, S, L, C = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "2,4,256,16384,4096").split(",")]
nu = L // C - 1
g = torch.Generator(device="cuda").manual_seed(0)
k = (torch.randn(B, L, K, S, device="cuda", generator=g) * 3).to(torch.bfloat16)
seg = torch.cumsum((torch.rand(B, L, device="cuda", generator=g) < 0.5 / C).long(), 1)
segc = seg.view(B, L // C, C)
kmask = (segc != segc[:, :, -1:]).reshape(B, L)
lse = torch.empty(B, K, nu, S, device="cuda")
op = AwmKeyLse(K, S, C)
f = cute.compile(op, fake(BFloat16), fake(Uint8, 1), fake(Float32), Int32(0), Int32(0), Int32(0),
                 cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
f(k.reshape(-1), kmask.view(torch.uint8).reshape(-1), lse.reshape(-1), B, L, nu)
torch.cuda.synchronize()
kf = k.float().view(B, L // C, C, K, S)[:, :nu].masked_fill(kmask.view(B, L // C, C, 1, 1)[:, :nu], float("-inf"))
ref = torch.logsumexp(kf, dim=2).permute(0, 2, 1, 3)  # B K nu S
print("lse max abs err", (lse - ref).abs().max().item(), "masked frac", kmask.float().mean().item())
sys.path.insert(0, ".")
from bench_kernel import gpu_time
args = (k.reshape(-1), kmask.view(torch.uint8).reshape(-1), lse.reshape(-1), B, L, nu)
t = gpu_time(lambda: f(*args))
nb = B * nu * C * K * S * 2
print(f"lse kernel {t:.1f} us  {nb / t / 1e6:.2f} TB/s")
