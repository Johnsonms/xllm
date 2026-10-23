import os
import sys
import torch

sys.path.insert(0, "/home/johnson/xllm/xllm/modules/fused_ops/cute")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timestep_decay_norm import timestep_decay_norm_fwd  # noqa: E402
from tsdn_ref import tsdn_ref  # noqa: E402

CASES = [  # B, L, H, G, dtype, bos_ratio, pad_ratio, config
    (2, 128, 256, 4, torch.float32, 0.0, 0.0, None),
    (3, 300, 256, 4, torch.float32, 0.02, 0.0, None),
    (2, 517, 512, 8, torch.float32, 0.02, 0.1, None),
    (2, 517, 512, 16, torch.float32, 0.01, 0.1, None),       # Dg=32 fp32 -> Gt=1 (128B)
    (2, 1000, 256, 32, torch.bfloat16, 0.01, 0.05, None),    # Dg=8 bf16 -> Gt=4
    (1, 9000, 1024, 16, torch.bfloat16, 0.0005, 0.0, None),  # >32 segments: multi-window look-back
    (2, 4100, 2048, 32, torch.bfloat16, 0.001, 0.01, None),
    (2, 700, 512, 8, torch.bfloat16, 0.01, 0.0, (2, 64, 128, 4, 2, 1)),
    (2, 700, 512, 8, torch.bfloat16, 0.01, 0.0, (1, 256, 256, 2, 2, 1)),
    (3, 2000, 512, 8, torch.bfloat16, 0.01, 0.05, (1, 64, 64, 4, 1, 1)),
]


def rel(a, r):
    return ((a.double() - r).abs().max() / r.abs().max().clamp_min(1e-30)).item()


def main():
    torch.manual_seed(0)
    worst = 0.0
    for B, L, H, G, dt, bos_r, pad_r, cfg in CASES:
        x = (torch.randn(B, L, H, device="cuda") * 2 + 0.5).to(dt)
        bos = torch.rand(B, L, device="cuda") < bos_r if bos_r > 0 else None
        pad = torch.rand(B, L, device="cuda") < pad_r if pad_r > 0 else None
        pc = torch.randint(0, 50, (B,), device="cuda")
        pm = (torch.randn(B, G, device="cuda") * 0.1).to(dt)
        pv = (torch.rand(B, G, device="cuda") * 0.1).to(dt)
        w = torch.randn(H, device="cuda").to(dt)
        b = torch.randn(H, device="cuda").to(dt)
        y, cnt, M, V, cm, cr, sc = timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, pad, G, 0.999, 0.9999, 1e-5, cfg)
        ry, rc, rM, rV, rcm, rcr = tsdn_ref(x, bos, pc, pm, pv, w, b, G, pad)
        errs = dict(y=rel(y, ry), cm=rel(cm, rcm), cr=rel(cr, rcr), M=rel(M, rM), V=rel(V, rV))
        cnt_ok = torch.equal(cnt, rc)
        tol = 2e-4 if dt == torch.float32 else 1e-2
        bad = [k for k, v in errs.items() if v > tol] + ([] if cnt_ok else ["count"])
        worst = max(worst, max(errs.values()) if dt == torch.float32 else 0)
        status = "OK  " if not bad else f"FAIL {bad}"
        print(f"{status} B{B} L{L} H{H} G{G} {str(dt)[6:]:8} cfg={cfg} " + " ".join(f"{k}={v:.1e}" for k, v in errs.items()))
    print("worst fp32 rel err", worst)


if __name__ == "__main__":
    main()
