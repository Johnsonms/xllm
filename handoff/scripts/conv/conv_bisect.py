import importlib.util, sys, torch
_s = importlib.util.spec_from_file_location("cute_conv_s", "/tmp/claude-243000396/-data-home-johnson-xllm/d3839bfb-7816-4668-9945-7c337f0c3f9e/scratchpad/sca/causal_conv_smem.py")
cv = importlib.util.module_from_spec(_s); _s.loader.exec_module(cv)
from cutlass import BFloat16, Float32, Int32, Uint8
for act in (False, True):
    for bos in (False, True):
        try:
            cv._compiled(("bisect_fwd", act, bos), lambda: cv.ConvFwdS(act, bos, 1024, 32), lambda: (
                cv._fake(BFloat16), cv._fake(Float32, 32), cv._fake(Uint8, 1), cv._fake(BFloat16), Int32(0), Int32(0), Int32(0)))
            print(f"fwd act={act} bos={bos}: ok", flush=True)
        except Exception as e:
            print(f"fwd act={act} bos={bos}: FAIL {str(e)[:100]}", flush=True)
        try:
            cv._compiled(("bisect_bwd", act, bos), lambda: cv.ConvBwdS(act, bos, 512, 32), lambda: (
                cv._fake(BFloat16), cv._fake(BFloat16), cv._fake(Float32, 32), cv._fake(Uint8, 1), cv._fake(BFloat16), cv._fake(Float32, 32), Int32(0), Int32(0), Int32(0)))
            print(f"bwd act={act} bos={bos}: ok", flush=True)
        except Exception as e:
            print(f"bwd act={act} bos={bos}: FAIL {str(e)[:100]}", flush=True)
