"""Phase 3 (out = q @ M_c, masked) against torch."""
import sys
import torch
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Uint8, Int32
sys.path.insert(0, "/home/johnson/xllm/xllm/modules/fused_ops/cute")
from adaptive_working_memory import AwmOutGemm


def fake(dtype, align=16):
    return cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(),), assumed_align=align)


def run(B, H, K, S, V, L, C, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    nc = L // C
    q = torch.softmax(torch.randn(B, L, H, S, device="cuda", generator=g) * 2, -1).to(torch.bfloat16)
    MT = (torch.randn(B, K, nc, V, S, device="cuda", generator=g)).to(torch.bfloat16)
    mask = torch.rand(B, L, device="cuda", generator=g) < 0.3
    out = torch.full((B, L, H, V), 7.0, device="cuda", dtype=torch.bfloat16)
    op = AwmOutGemm(H, K, S, V, C)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    num_tiles = (V // op.VT) * B * K * nc * (H // K) * (C // 128)
    ctas = min(sms, num_tiles)
    compiled = cute.compile(op, fake(BFloat16), fake(BFloat16), fake(BFloat16), fake(Uint8, 1), Int32(0), Int32(0),
                            Int32(0), cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
    compiled(q.reshape(-1), MT.reshape(-1), out.reshape(-1), mask.view(torch.uint8).reshape(-1), B, L, ctas)
    torch.cuda.synchronize()
    Mh = MT.float().repeat_interleave(H // K, dim=1)  # B H nc V S
    qc = q.float().view(B, nc, C, H, S)
    ref = torch.einsum("bnchs,bhnvs->bnchv", qc, Mh).reshape(B, L, H, V)
    ref = ref.masked_fill(mask.view(B, L, 1, 1), 0.0)
    err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
    print(f"B{B} H{H} K{K} S{S} V{V} L{L} C{C} ctas={ctas}: rel err {err:.2e} masked_zero={bool((out.float()[mask] == 0).all())}")
    return compiled, (q, MT, out, mask, ctas)


if __name__ == "__main__":
    shapes = [tuple(int(x) for x in a.split(",")) for a in sys.argv[1:]] or [(1, 2, 1, 64, 128, 256, 128)]
    for sh in shapes:
        run(*sh)
