"""Phase-by-phase check of the CuTe backward kernels against the FP64 phased spec (bf16 inputs)."""
import importlib.util, sys, os
sys.path.insert(0, "/home/johnson/xllm/tests")
import torch
from test_cute_adaptive_working_memory import awm, awm_ref, make_inputs, rel_err
_p = "/home/johnson/xllm/xllm/modules/fused_ops/cute/adaptive_working_memory_bwd.py"
_s = importlib.util.spec_from_file_location("cute_awm_bwd", _p)
bwd = importlib.util.module_from_spec(_s)
_s.loader.exec_module(bwd)

CASES = [(2, 4, 2, 128, 128, 1024, 256, 0.0), (2, 4, 2, 128, 128, 1024, 256, 0.6), (1, 4, 2, 256, 256, 2048, 512, 1.5),
         (2, 8, 4, 256, 256, 16384, 4096, 0.3), (1, 32, 8, 128, 128, 16384, 4096, 0.3)]
phases = sys.argv[1:] or ["3"]
for (B, H, K, S, V, L, C, seg_rate) in CASES:
    q, qk, k, v, memory, lnt, pqk, pk, pv, seg, cnt = make_inputs(B, H, K, S, V, L, C, False, seg_rate)
    g = torch.Generator(device="cuda").manual_seed(1)
    dout = torch.randn(B, L, H, V, device="cuda", generator=g).to(torch.bfloat16).transpose(1, 2)  # [B, H, L, V] view
    nc = L // C
    # spec in FP64 on the same bf16 inputs
    with torch.no_grad():
        out_r, M_r, lt_r, saved = awm_ref.awm_fwd_saved(q.double(), qk.double(), k.double(), v.double(), C, None, None, None, None, None, seg, None)
        inter = {}
        dq_r, dqk_r, dk_r, dv_r = awm_ref.awm_bwd_phased(dout.double(), torch.zeros_like(M_r), torch.zeros_like(lt_r),
                                                         q.double(), qk.double(), k.double(), v.double(), C, saved, inter)
    w = awm._phases12(qk, k, v, seg, None, None, None, None, None, None, C, save=True)
    tag = f"B{B} H{H} K{K} S{S} V{V} L{L} C{C} seg={seg_rate}"
    if "3" in phases:
        dq, out_mask, dMo = bwd._phase3_bwd(awm._blhs(dout), awm._blhs(q), w, H, K, S, V, C, False, seg is not None)
        torch.cuda.synchronize()
        # dq with the kernel's own bf16 M sequence (the spec uses the fp64 one)
        dq_r2 = torch.einsum("bhncv,bhnsv->bhncs", dout.double().view(B, H, nc, C, V).masked_fill(saved["out_mask"][:, None, :, :, None], 0.0),
                             w["Mseq"].double().repeat_interleave(H // K, 1)).reshape(B, H, L, S)
        print(f"{tag}: mask_ok={torch.equal(out_mask.view(B, nc, C), saved['out_mask'])} dq {rel_err(dq.transpose(1, 2), inter['dq'].reshape(B, H, L, S)):.1e} "
              f"(vs bf16-M spec {rel_err(dq.transpose(1, 2), dq_r2):.1e}) dMo {rel_err(dMo, inter['dMo']):.1e}")
    if "2" in phases:
        if "3" not in phases:
            dq, out_mask, dMo = bwd._phase3_bwd(awm._blhs(dout), awm._blhs(q), w, H, K, S, V, C, False, seg is not None)
        dmem = torch.randn(B, K, S, V, device="cuda") if "dmem" in phases else None
        dlnt = torch.randn(B, K, S, device="cuda") if "dmem" in phases else None
        if dmem is not None:
            inter = {}
            awm_ref.awm_bwd_phased(dout.double(), dmem.double(), dlnt.double(), q.double(), qk.double(), k.double(), v.double(), C, saved, inter)
        dB16, dA16, delta, dlse, gmem, glnt = bwd._phase2_bwd(w, dMo, dmem, dlnt, K, S, V, C, False, seg is not None)
        torch.cuda.synchronize()
        print(f"{tag}: dB {rel_err(dB16, inter['dB']):.1e} dA {rel_err(dA16, inter['dA']):.1e} delta {rel_err(delta, inter['delta']):.1e} "
              f"dlse {rel_err(dlse, inter['dlse']):.1e} G0 {rel_err(gmem, inter['G']):.1e} g0 {rel_err(glnt, inter['g']):.1e}")
    if "1" in phases:
        if "2" not in phases:
            dq, out_mask, dMo = bwd._phase3_bwd(awm._blhs(dout), awm._blhs(q), w, H, K, S, V, C, False, seg is not None)
            dB16, dA16, delta, dlse, gmem, glnt = bwd._phase2_bwd(w, dMo, None, None, K, S, V, C, False, seg is not None)
        dqk, dk, dv = bwd._phase1_bwd(w, dA16, dB16, delta, dlse, awm._blhs(qk), awm._blhs(k), awm._blhs(v), K, S, V, C, seg is not None)
        torch.cuda.synchronize()
        print(f"{tag}: dqk {rel_err(dqk.transpose(1, 2), dqk_r):.1e} dk {rel_err(dk.transpose(1, 2), dk_r):.1e} dv {rel_err(dv.transpose(1, 2), dv_r):.1e}")
