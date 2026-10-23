import sys, torch
sys.path.insert(0, "cubext/build")
import tsdn_cub
from tsdn_ref import tsdn_ref
torch.manual_seed(0)
for (B, L, H, G, padr) in [(3, 300, 256, 4, 0.0), (2, 517, 512, 8, 0.1)]:
    dt = torch.float32
    x = torch.randn(B, L, H, device="cuda") * 2 + 0.5
    bos = torch.rand(B, L, device="cuda") < 0.02
    pad = (torch.rand(B, L, device="cuda") < padr) if padr > 0 else None
    pc = torch.randint(0, 50, (B,), device="cuda")
    pm = torch.randn(B, G, device="cuda") * 0.1
    pv = torch.rand(B, G, device="cuda") * 0.1
    w = torch.randn(H, device="cuda"); b = torch.randn(H, device="cuda")
    dy = torch.randn(B, L, H, device="cuda"); mg = torch.randn(B, G, device="cuda"); vg = torch.randn(B, G, device="cuda")
    y, cnt, M, V, cm, cr = tsdn_cub.group_timestep_decay_norm_cub_fwd(x, bos, pc, pm, pv, w, b, pad, G, 0.999, 0.9999, 1e-5)
    leaves = [t.double().requires_grad_() for t in (x, pm, pv, w, b)]
    ry, rc, rM, rV, rcm, rcr = tsdn_ref(leaves[0], bos, pc, leaves[1], leaves[2], leaves[3], leaves[4], G, pad)
    loss = (ry * dy.double()).sum() + (rM * mg.double()).sum() + (rV * vg.double()).sum()
    loss.backward()
    gx, gpm, gpv, gw, gb = tsdn_cub.group_timestep_decay_norm_cub_bwd(dy, mg, vg, x, pc, bos, cm, cr, w, pad, G, 0.999, 0.9999)
    rel = lambda a, r: ((a.double() - r).abs().max() / r.abs().max()).item()
    print(f"B{B} L{L} H{H} G{G} pad{padr}: y {rel(y, ry):.1e} cm {rel(cm, rcm):.1e} cr {rel(cr, rcr):.1e} M {rel(M, rM):.1e} V {rel(V, rV):.1e} cnt {(cnt==rc).all().item()}"
          f" | dx {rel(gx, leaves[0].grad):.1e} dpm {rel(gpm, leaves[1].grad):.1e} dpv {rel(gpv, leaves[2].grad):.1e} dw {rel(gw, leaves[3].grad):.1e} db {rel(gb, leaves[4].grad):.1e}")
