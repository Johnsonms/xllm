import importlib.util, sys
sys.path.insert(0, "/home/johnson/xllm/tests")
import torch
from test_cute_adaptive_working_memory import awm, awm_ref, make_inputs, rel_err
for (B, H, K, S, V, L, C, seg_rate) in [(2, 4, 2, 128, 128, 1024, 256, 0.0), (2, 4, 2, 128, 128, 1024, 256, 0.6),
                                        (1, 4, 2, 256, 256, 2048, 512, 1.5), (1, 2, 2, 256, 1024, 4096, 1024, 0.5),
                                        (1, 4, 4, 256, 2048, 16384, 4096, 0.3)]:
    q, qk, k, v, memory, lnt, pqk, pk, pv, seg, cnt = make_inputs(B, H, K, S, V, L, C, False, seg_rate)
    w = awm._phases12(qk, k, v, seg, None, None, None, None, None, None, C, save=True)
    _, _, _, sv = awm_ref.awm_fwd_saved(q.double(), qk.double(), k.double(), v.double(), C, None, None, None, None, None, seg, None)
    Dref = torch.stack([sv["Bm"][:, :, u] - sv["A"][:, :, u] @ sv["Mpre"][u] - sv["Mpre"][u] for u in range(L // C - 1)], 2)
    Rref = torch.stack(sv["rs"], 2)
    torch.cuda.synchronize()
    print(f"B{B} K{K} S{S} V{V} L{L} C{C} seg={seg_rate}: Mseq {rel_err(w['Mseq'], sv['Mseq']):.1e} D {rel_err(w['D'], Dref):.1e} "
          f"R {rel_err(w['R'], Rref):.1e} (ref r range {Rref.min().item():.3f}..{Rref.max().item():.3f}) MT {rel_err(w['MT'], sv['Mseq'].transpose(-1, -2)):.1e}")
