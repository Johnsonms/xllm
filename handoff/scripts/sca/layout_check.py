import cutlass, cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32
from cutlass.cute.nvgpu import tcgen05

@cute.jit
def show():
    tm_k = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(BFloat16, Float32, (128, 64, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
    tm_mn = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(BFloat16, Float32, (128, 256, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
    lk = sm100_utils.make_smem_layout_b(tm_k, (128, 64, 256), BFloat16, 2)
    lmn = sm100_utils.make_smem_layout_b(tm_mn, (128, 256, 64), BFloat16, 2)
    cute.printf("K-major  : {}", lk)
    cute.printf("MN-major : {}", lmn)
show()
