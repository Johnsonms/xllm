import cutlass, cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32
from cutlass.cute.nvgpu import tcgen05

@cute.jit
def show():
    tm_k = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(BFloat16, Float32, (128, 64, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
    tm_mn = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(BFloat16, Float32, (128, 128, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
    lk = sm100_utils.make_smem_layout_b(tm_k, (128, 64, 128), BFloat16, 2)
    lmn = sm100_utils.make_smem_layout_b(tm_mn, (128, 128, 64), BFloat16, 2)
    cute.printf("dO K-major  B (N=64,K=128): {}", lk)
    cute.printf("dO MN-major B (N=128,K=64): {}", lmn)
    pl = sm100_utils.make_smem_layout_a(tm_mn, (128, 128, 64), BFloat16, 1)
    cute.printf("P^T A (TMEM) layout: {}", pl)
show()
