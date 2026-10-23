"""Phase 1 (lse + A|B chunk GEMM) against torch."""
import sys
import torch
import cutlass.cute as cute
from cutlass import BFloat16, Uint8, Int32, Float32
sys.path.insert(0, "/home/johnson/xllm/xllm/modules/fused_ops/cute")
from adaptive_working_memory import AwmKeyLse, AwmChunkGemm


def fake(dtype, align=16):
    return cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(),), assumed_align=align)


def fstream():
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)


def run(B, K, S, V, L, C, splits, seed=0, seg_rate=0.5):
    nu = L // C - 1
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = (torch.randn(B, L, K, S, device="cuda", generator=g) * 2).to(torch.bfloat16)
    qk = torch.softmax(k.float(), -1).to(torch.bfloat16)
    v = torch.randn(B, L, K, V, device="cuda", generator=g).to(torch.bfloat16)
    seg = torch.cumsum((torch.rand(B, L, device="cuda", generator=g) < seg_rate / C).long(), 1)
    segc = seg.view(B, L // C, C)
    kmask = (segc != segc[:, :, -1:]).reshape(B, L).view(torch.uint8)
    lse = torch.empty(B, K, nu, S, device="cuda")
    AB = torch.zeros(B, K, nu, S, S + V, device="cuda")
    fl = cute.compile(AwmKeyLse(K, S, C), fake(BFloat16), fake(Uint8, 1), fake(Float32), Int32(0), Int32(0), Int32(0),
                      fstream(), options="--enable-tvm-ffi")
    print('lse compiled', flush=True)
    op = AwmChunkGemm(K, S, V, C, splits)
    fg = cute.compile(op, fake(BFloat16), fake(BFloat16), fake(BFloat16), fake(Uint8, 1), fake(Float32), fake(Float32),
                      Int32(0), Int32(0), Int32(0), fstream(), options="--enable-tvm-ffi")
    print('gemm compiled', flush=True)
    flat = lambda t: t.reshape(-1)
    fl(flat(k), flat(kmask), flat(lse), B, L, nu)
    torch.cuda.synchronize()
    print('lse ran', flush=True)
    import os
    if os.environ.get("NORUN"): return
    fg(flat(k), flat(qk), flat(v), flat(kmask), flat(lse), flat(AB), B, L, nu)
    torch.cuda.synchronize()
    print('gemm ran', flush=True)
    kf = k.double().view(B, L // C, C, K, S)[:, :nu].masked_fill(kmask.bool().view(B, L // C, C, 1, 1)[:, :nu], float("-inf"))
    P = torch.exp(kf - torch.logsumexp(kf, dim=2, keepdim=True))  # B nu C K S
    X = torch.cat([qk.double(), v.double()], -1).view(B, L // C, C, K, S + V)[:, :nu]
    ref = torch.einsum("bnckS,bnckN->bknSN", P, X)
    err = ((AB.double() - ref).abs().max() / ref.abs().max()).item()
    print(f"B{B} K{K} S{S} V{V} L{L} C{C} splits={splits} tiling(NS,nsub,nb)={op.NS, op.nsub, op.n_blocks} "
          f"grid={B * K * nu * op.m_blocks * op.n_blocks * splits}: rel err {err:.2e}")
    return fl, fg, (k, qk, v, kmask, lse, AB, nu)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        run(*(int(x) for x in a.split(",")))
