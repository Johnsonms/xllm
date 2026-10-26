"""DeepGEMM sm90 fp8 GEMM vs xllm's cuBLASLt blockwise path (fp8_sm90.mm) at the gekko shapes, one GPU.

For each linear (tokens T, out N, in K) the three GEMMs of a training step:
  fwd   Y  = X W^T    (T, N, K)  A = X  1x128,  B = W  128x128
  dgrad dX = dY W     (T, K, N)  A = dY 1x128,  B = W^T 128x128
  wgrad dW = dY^T X   (N, K, T)  cuBLAS default WGRAD_TILE=dy: A = dY^T 128x128, B = X^T 1x128;
                                 DeepGEMM sm90 needs A 1x128, B 128x128, i.e. WGRAD_TILE=x
Prints us and TFLOP/s per GEMM and the relative difference DeepGEMM vs cuBLAS on the same operands.
usage: PYTHONPATH=<xllm root> python dg_bench.py   (deep_gemm installed in the venv)
"""
import sys, types
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
import torch
import deep_gemm
from xllm.modules.fused_ops import fp8_sm90 as f8

SHAPES = {  # name: (T, N, K)
    "0.9B fc1|fc3": (32768, 8192, 1536), "0.9B fc2": (32768, 1536, 4096), "0.9B q": (32768, 2048, 1536),
    "4B fc1|fc3": (8192, 18432, 2560), "4B fc2": (8192, 2560, 9216), "4B q": (8192, 4096, 2560),
    "7.1B fc1|fc3": (8192, 16384, 4096), "7.1B fc2": (8192, 4096, 8192), "7.1B v": (8192, 8192, 4096),
    "7.1B wo": (8192, 4096, 8192),
}


def bench(f, n=50):
    for _ in range(5): f()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(n): f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1e3


def q(x, tile):
    r, _ = f8.quantize(x, colwise=False, kind="weight" if tile else "grad")
    return r


def dg(a, b, d, c=None):
    deep_gemm.fp8_gemm_nt((a.data, a.scale), (b.data, b.scale), d, c=c)


def one(M, N, K, a_tile_cublas):
    a_bf = torch.randn(M, K, device="cuda").bfloat16()
    b_bf = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
    a1, bt = q(a_bf, False), q(b_bf, True)          # DeepGEMM layout (and cuBLAS fwd / dgrad layout)
    fl = 2 * M * N * K
    if a_tile_cublas:                               # cuBLAS wgrad default: A tiled, B 1x128
        ac, bc = q(a_bf, True), q(b_bf, False)
    else:
        ac, bc = a1, bt
    t_cb = bench(lambda: f8.mm(ac, bc))
    t_cb_same = bench(lambda: f8.mm(a1, bt)) if a_tile_cublas else t_cb
    ref = f8.mm(a1, bt).float()
    d = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    dg(a1, bt, d)
    # vs a dequantised fp32 reference too (the cuBLAS result could share bits by accident)
    if True:
        ref32 = f8.dequantize(a1) @ f8.dequantize(bt).t()
        e_dg = ((d.float() - ref32).norm() / ref32.norm()).item(); e_cb = ((ref - ref32).norm() / ref32.norm()).item()
        print(f"   vs fp32: DeepGEMM {e_dg:.1e} cuBLAS {e_cb:.1e}, bitwise equal {torch.equal(d.float(), ref)}", flush=True)
    diff = ((d.float() - ref).norm() / ref.norm()).item()
    t_dg = bench(lambda: dg(a1, bt, d))
    # residual epilogue: cuBLAS ext (C matrix) vs DeepGEMM c= (if supported for bf16 out)
    res = torch.randn(M, N, device="cuda").bfloat16()
    t_cb_res = bench(lambda: f8.mm(a1, bt, residual=res))
    try:
        dc = torch.empty_like(d); dg(a1, bt, dc, c=res)
        rdiff = ((dc.float() - (ref + res.float())).norm() / (ref + res.float()).norm()).item()
        t_dg_res = bench(lambda: dg(a1, bt, dc, c=res)); res_s = f"{t_dg_res:6.0f}us (diff {rdiff:.1e})"
    except Exception as ex:
        res_s = f"unsupported ({str(ex).splitlines()[0][:60]})"
    return t_cb, t_cb_same, t_dg, diff, fl, t_cb_res, res_s


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"{torch.cuda.get_device_name()}  deep_gemm {getattr(deep_gemm, '__version__', '?')}")
    tot = {"cublas": 0.0, "dg": 0.0}
    for name, (T, N, K) in SHAPES.items():
        for role, (M, NN, KK), atile in (("fwd", (T, N, K), False), ("dgrad", (T, K, N), False), ("wgrad", (N, K, T), True)):
            t_cb, t_cb_same, t_dg, diff, fl, t_cb_res, res_s = one(M, NN, KK, atile)
            tot["cublas"] += t_cb; tot["dg"] += t_dg
            extra = f" (cuBLAS with DG's tiling {t_cb_same:.0f}us)" if atile else ""
            print(f"{name:13s} {role:5s} {M:6d}x{NN:6d}x{KK:6d}: cuBLAS {t_cb:6.0f}us {fl/t_cb/1e6:5.0f}TF{extra} | "
                  f"DeepGEMM {t_dg:6.0f}us {fl/t_dg/1e6:5.0f}TF  x{t_cb/t_dg:.2f}  diff {diff:.1e}"
                  + (f" | +residual cuBLAS {t_cb_res:.0f}us DG {res_s}" if role != "wgrad" else ""), flush=True)
    print(f"sum over GEMMs: cuBLAS {tot['cublas']/1e3:.2f} ms, DeepGEMM {tot['dg']/1e3:.2f} ms (x{tot['cublas']/tot['dg']:.2f})")
