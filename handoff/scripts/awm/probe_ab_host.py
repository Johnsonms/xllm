import sys
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05

STEP = int(sys.argv[1])


@cute.jit
def host(tK: cute.Tensor, L: Int32, B: Int32):
    K, S = 1, 128
    mK = cute.make_tensor(tK.iterator, cute.make_layout((S, L, K, B), stride=(1, K * S, S, L * K * S)))
    op = tcgen05.MmaF16BF16Op(BFloat16, Float32, (128, 128, 16), tcgen05.CtaGroup.ONE,
                              tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.MN, tcgen05.OperandMajorMode.MN)
    tiled_mma = cute.make_tiled_mma(op)
    tiler = (128, 128, 32)
    a_layout = sm100_utils.make_smem_layout_a(tiled_mma, tiler, BFloat16, 4)
    print("a_layout", a_layout)
    b_layout = sm100_utils.make_smem_layout_b(tiled_mma, tiler, BFloat16, 8)
    print("b_layout", b_layout)
    tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    if cutlass.const_expr(STEP >= 1):
        tma_k, mK_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mK, cute.slice_(a_layout, (None, None, None, 0)), tiler, tiled_mma)
        print("tma A ok", mK_t)
    if cutlass.const_expr(STEP >= 2):
        tma_b, mB_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mK, cute.slice_(b_layout, (None, None, None, 0)), tiler, tiled_mma)
        print("tma B ok", mB_t)


import torch
t = torch.zeros(1 << 20, dtype=torch.bfloat16, device="cuda")
from cutlass.cute.runtime import from_dlpack
cute.compile(host, from_dlpack(t, assumed_align=16), Int32(4096), Int32(1))
print("done")
