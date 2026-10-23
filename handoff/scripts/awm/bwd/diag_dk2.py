import importlib.util, sys
sys.path.insert(0, "/home/johnson/xllm/tests")
import torch
from test_cute_adaptive_working_memory import awm, awm_ref, make_inputs, rel_err
_s = importlib.util.spec_from_file_location("cute_awm_bwd", "/home/johnson/xllm/xllm/modules/fused_ops/cute/adaptive_working_memory_bwd.py")
bwd = importlib.util.module_from_spec(_s); _s.loader.exec_module(bwd)
mode = sys.argv[1]
if mode == "nonpersistent":
    bwd._sm_count = lambda dev: 1 << 20
elif mode == "stages2":
    orig = bwd.AwmChunkBwd.__init__
    def init(self, *a, **kw):
        orig(self, *a, **kw); self.a_stages = 2; self.b_stages = 2
    bwd.AwmChunkBwd.__init__ = init
for (B, H, K, S, V, L, C, seg_rate) in [(1, 32, 8, 128, 128, 16384, 4096, 0.3), (2, 8, 4, 256, 256, 16384, 4096, 0.3)]:
    q, qk, k, v, _, _, _, _, _, seg, _ = make_inputs(B, H, K, S, V, L, C, False, seg_rate)
    g = torch.Generator(device="cuda").manual_seed(1)
    dout = torch.randn(B, L, H, V, device="cuda", generator=g).to(torch.bfloat16).transpose(1, 2)
    with torch.no_grad():
        out_r, M_r, lt_r, saved = awm_ref.awm_fwd_saved(q.double(), qk.double(), k.double(), v.double(), C, None, None, None, None, None, seg, None)
        dq_r, dqk_r, dk_r, dv_r = awm_ref.awm_bwd_phased(dout.double(), torch.zeros_like(M_r), torch.zeros_like(lt_r), q.double(), qk.double(), k.double(), v.double(), C, saved)
    for rep in range(3):
        dq, dqk, dk, dv, _, _ = bwd.adaptive_working_memory_bwd(dout, None, None, q, qk, k, v, C, seg)
        torch.cuda.synchronize()
        print(f"{mode} S{S} rep{rep}: dqk {rel_err(dqk, dqk_r):.1e} dk {rel_err(dk, dk_r):.1e} dv {rel_err(dv, dv_r):.1e}")
