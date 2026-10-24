import cutlass, cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32
from cutlass.cute.nvgpu import tcgen05
@cute.jit
def show():
    tm_s = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(BFloat16, Float32, (128, 32, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
    tm_dk = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(BFloat16, Float32, (128, 128, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
    tm_dv = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(BFloat16, Float32, (128, 128, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
    cute.printf("K TMEM A (128,32,128): {}", sm100_utils.make_smem_layout_a(tm_s, (128, 32, 128), BFloat16, 1))
    cute.printf("Q smem B K-major (128,32,128) x4: {}", sm100_utils.make_smem_layout_b(tm_s, (128, 32, 128), BFloat16, 4))
    cute.printf("Q smem B MN-major (128,128,32) x4: {}", sm100_utils.make_smem_layout_b(tm_dk, (128, 128, 32), BFloat16, 4))
    cute.printf("dO smem B K-major (128,32,128) x3: {}", sm100_utils.make_smem_layout_b(tm_s, (128, 32, 128), BFloat16, 3))
    cute.printf("dO smem B MN-major (128,128,32) x3: {}", sm100_utils.make_smem_layout_b(tm_dv, (128, 128, 32), BFloat16, 3))
    cute.printf("dS smem A K-major (128,128,32) x2: {}", sm100_utils.make_smem_layout_a(tm_dk, (128, 128, 32), BFloat16, 2))
    cute.printf("P TMEM A (128,128,32): {}", sm100_utils.make_smem_layout_a(tm_dv, (128, 128, 32), BFloat16, 1))
show()
