import importlib.util, sys, time, types, torch
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
sys.path.insert(0, "/home/johnson/xllm"); sys.path.insert(0, "/home/johnson/scratch/xext")
_s = importlib.util.spec_from_file_location("cute_conv", "/tmp/claude-243000396/-data-home-johnson-xllm/d3839bfb-7816-4668-9945-7c337f0c3f9e/scratchpad/sca/causal_conv_draft.py")
cv = importlib.util.module_from_spec(_s); _s.loader.exec_module(cv)
from xllm.modules.fused_ops.conv.triton.causal_conv1d_reference import causal_conv1d_reference_fwd, causal_conv1d_reference_bwd
from xllm.modules.fused_ops.conv.triton_causal_conv1d import _triton_causal_conv1d_fwd, _triton_causal_conv1d_bwd
torch.manual_seed(0)
def timeit(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
for (B, L, D, act, bosp) in [(2, 16384, 2048, None, 0.001), (2, 16384, 1024, "silu", 0.001), (1, 16384, 8192, "silu", 0.0005), (3, 100, 64, "silu", 0.1), (1, 35, 8, None, 0.0), (2, 70, 16, None, 0.3)]:
    x = torch.randn(B, L, D, device="cuda").to(torch.bfloat16)
    w = torch.softmax(torch.randn(D, 4, device="cuda"), -1)  # fp32 normalized weights
    bos = (torch.rand(B, L, device="cuda") < bosp); bos[:, 0] = True
    if bosp == 0.0: bos = None
    g = torch.randn_like(x)
    o_ref, _ = causal_conv1d_reference_fwd(x, w, None, None, bos, act)
    o = cv.conv_fwd(x, w, bos, act)
    o_tr, _, _ = _triton_causal_conv1d_fwd(x, w, None, None, bos, False, act)
    dx_ref, _, dw_ref, _ = causal_conv1d_reference_bwd(g, None, x, w, None, None, bos, act)
    dx, dw = cv.conv_bwd(g, x, w, bos, act)
    dx_tr, _, dw_tr, _ = _triton_causal_conv1d_bwd(g, None, x, w, None, None, bos, act)
    print(f"B{B} L{L} D{D} act={act} bos={bosp}: out err {(o.float()-o_ref.float()).abs().max().item():.2e} (triton {(o_tr.float()-o_ref.float()).abs().max().item():.2e}) | dx err {(dx.float()-dx_ref.float()).abs().max().item():.2e} (triton {(dx_tr.float()-dx_ref.float()).abs().max().item():.2e}, max {dx_ref.abs().max().item():.1f}) dw err {(dw-dw_ref).abs().max().item():.2e} (triton {(dw_tr-dw_ref).abs().max().item():.2e}, max {dw_ref.abs().max().item():.0f})")
    if L >= 1000:
        print(f"   fwd triton {timeit(lambda: _triton_causal_conv1d_fwd(x, w, None, None, bos, False, act)):.0f} cute {timeit(lambda: cv.conv_fwd(x, w, bos, act)):.0f} us | bwd triton {timeit(lambda: _triton_causal_conv1d_bwd(g, None, x, w, None, None, bos, act)):.0f} cute {timeit(lambda: cv.conv_bwd(g, x, w, bos, act)):.0f} us")
