"""TimestepDecayNorm(backend='cute') end to end through autograd, with the C++ extension stubbed out."""
import sys, types, importlib.util, torch
for name in ["xllm_extension", "xllm_extension.ops"]:
    m = types.ModuleType(name)
    for op in ["group_timestep_decay_norm_fwd", "group_timestep_decay_norm_cub_fwd", "group_timestep_decay_norm_bwd", "group_timestep_decay_norm_cub_bwd"]:
        setattr(m, op, None)
    sys.modules[name] = m
dist = types.ModuleType("xllm.distributed"); dist.get_model_parallel_world_size = lambda: 1
dutil = types.ModuleType("xllm.distributed.utils"); dutil.divide_and_check_no_remainder = lambda a, b: a // b
xllm_pkg = types.ModuleType("xllm"); xllm_pkg.__path__ = ["/home/johnson/xllm/xllm"]
mods_pkg = types.ModuleType("xllm.modules"); mods_pkg.__path__ = ["/home/johnson/xllm/xllm/modules"]
fo_pkg = types.ModuleType("xllm.modules.fused_ops"); fo_pkg.__path__ = ["/home/johnson/xllm/xllm/modules/fused_ops"]
sys.modules.update({"xllm": xllm_pkg, "xllm.distributed": dist, "xllm.distributed.utils": dutil, "xllm.modules": mods_pkg, "xllm.modules.fused_ops": fo_pkg})
spec = importlib.util.spec_from_file_location("xllm.modules.timestep_decay_norm", "/home/johnson/xllm/xllm/modules/timestep_decay_norm.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
spec = importlib.util.spec_from_file_location("tt", "/home/johnson/xllm/tests/normalization/test_cute_timestep_decay_norm.py")
tt = importlib.util.module_from_spec(spec); spec.loader.exec_module(tt)

torch.manual_seed(0)
B, L, H, G = 2, 1500, 512, 8
for dt in [torch.float32, torch.bfloat16]:
    norm = mod.TimestepDecayNorm(H, G, backend="cute").cuda().to(dt)
    with torch.no_grad():
        norm.weight.normal_(0, 0.2); norm.bias.normal_(0, 0.2)
    x = torch.randn(B, L, H, device="cuda", dtype=dt, requires_grad=True)
    bos = torch.rand(B, L, device="cuda") < 0.01
    pad = torch.rand(B, L, device="cuda") < 0.03
    y, count, mean, var = norm(x, bos, padding_mask=pad)
    dy = torch.randn_like(y); mg = torch.randn_like(mean); vg = torch.randn_like(var)
    ((y.float() * dy.float()).sum() + (mean.float() * mg.float()).sum() + (var.float() * vg.float()).sum()).backward()
    # reference
    xr = x.detach().double().requires_grad_(); wr = norm.weight.detach().double().requires_grad_(); br = norm.bias.detach().double().requires_grad_()
    pc = torch.zeros(B, dtype=torch.int64, device="cuda"); pm = torch.zeros(B, G, device="cuda", dtype=torch.float64); pv = torch.zeros_like(pm)
    ry, rc, rM, rV, *_ = tt.reference_fwd(xr, bos, pc, pm, pv, wr + 1.0, br, G, pad)
    ((ry * dy.double()).sum() + (rM * mg.double()).sum() + (rV * vg.double()).sum()).backward()
    e = lambda a, r: tt.rel_err(a, r)
    print(f"{str(dt)[6:]:8} y={e(y, ry):.1e} mean={e(mean, rM):.1e} count_ok={torch.equal(count, rc)} | dx={e(x.grad, xr.grad):.1e} dw={e(norm.weight.grad, wr.grad):.1e} db={e(norm.bias.grad, br.grad):.1e}")
# step counts recomputed from prev_count + masks == forward's
from xllm.modules.fused_ops.cute.timestep_decay_norm import step_counts, timestep_decay_norm_fwd
x2, bos2, pc2, pm2, pv2, w2, b2, pad2 = tt.make_inputs(3, 2000, 256, 4, torch.float32, 0.01, 0.05)
sc = timestep_decay_norm_fwd(x2, bos2, pc2, pm2, pv2, w2, b2, pad2, 4, tt.BETA1, tt.BETA2, tt.EPS)[6]
print("step_counts matches forward:", torch.equal(sc, step_counts(bos2, pad2, pc2, 2000)))
