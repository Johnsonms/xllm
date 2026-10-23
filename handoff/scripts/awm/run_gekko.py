"""Single-GPU Gekko training step with random multi-segment tokens: records the shapes and arguments of the
AWM fused ops and their GPU time share.

usage: python run_gekko.py <preset> <batch> <seq_len> [fused_block=1] [steps=3]
"""
import os
import sys
import time
from collections import defaultdict

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(S, "xext"))
sys.path.insert(0, "/home/johnson/xllm")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29533")

import torch
import torch.distributed as dist
import types
# Gekko's sliding chunk attention uses the 'swift' backend (xllm_extension); flash-attn is only imported
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))

preset, bsz, seq_len = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
fused = (sys.argv[4] if len(sys.argv) > 4 else "1") == "1"
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 3

dist.init_process_group("nccl", world_size=1, rank=0)
torch.cuda.set_device(0)
from xllm.distributed import initialize_model_parallel  # noqa: E402
initialize_model_parallel(1, 1, 1800)

import xllm.modules.fused_ops  # noqa: E402,F401
awm_mod = sys.modules['xllm.modules.fused_ops.adaptive_working_memory']
import xllm.models.fused_blocks.gekko.gekko as fused_gekko  # noqa: E402
import xllm.modules.adaptive_working_memory as awm_module  # noqa: E402
from xllm.config import ModelConf  # noqa: E402,F401
from xllm.models.gekko import Gekko  # noqa: E402
from xllm.configuration import ConfStore  # noqa: E402

calls = defaultdict(list)
gpu_ms = defaultdict(float)


def describe(v):
    if isinstance(v, torch.Tensor):
        return f"{tuple(v.shape)}:{str(v.dtype).replace('torch.', '')}"
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(describe(x) for x in v) + ")"
    return repr(v)


def wrap(name, fn):
    def inner(*args, **kwargs):
        if len(calls[name]) < 2:
            calls[name].append([describe(a) for a in args] + [f"{k}={describe(v)}" for k, v in kwargs.items()])
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        out = fn(*args, **kwargs)
        e1.record()
        e1.synchronize()
        gpu_ms[name] += e0.elapsed_time(e1)
        return out
    return inner


USE_CUTE = os.environ.get("AWM_CUTE") == "1"
if USE_CUTE:
    sys.path.insert(0, "/home/johnson/xllm/xllm/modules/fused_ops/cute")
    import adaptive_working_memory as cute_awm  # noqa: E402
    awm_mod.adaptive_working_memory_fwd = cute_awm.adaptive_working_memory_fwd
    awm_mod._adaptive_working_memory_fwd = cute_awm.adaptive_working_memory_fwd
for name in ["adaptive_working_memory_fwd", "adaptive_working_memory_accum_fwd", "adaptive_working_memory_bwd"]:
    w = wrap(name, getattr(awm_mod, name))
    setattr(awm_mod, name, w)
    if hasattr(fused_gekko, name):
        setattr(fused_gekko, name, w)
for name in ["_adaptive_working_memory_fwd", "_adaptive_working_memory_accum_fwd", "_adaptive_working_memory_bwd"]:
    setattr(awm_mod, name, wrap(name, getattr(awm_mod, name)))
awm_module.adaptive_working_memory = awm_mod.AdaptiveWorkingMemoryFunc.apply


class Tok:
    bos_id, eos_id, pad_id, vocab_size, n_words = 1, 2, 0, 32000, 32000


cfg = ConfStore[preset]
cfg.vocab_size = Tok.vocab_size
cfg.fused_block = fused
cfg.awm_backend = os.environ.get("AWM_BACKEND", "torch")
torch.manual_seed(0)
model = Gekko(cfg, Tok()).cuda().to(torch.bfloat16)
model.train()
print(f"{preset}: H={cfg.num_heads} K={cfg.num_kv_heads} S={cfg.head_dim} V={cfg.v_head_dim} C={cfg.chunk_size} "
      f"ortho={cfg.awm_orthogonal_update} fused_block={fused} layers={cfg.num_layers} B={bsz} L={seq_len}")

g = torch.Generator(device="cuda").manual_seed(0)
for step in range(steps):
    tokens = torch.randint(3, Tok.vocab_size, (bsz, seq_len + 1), device="cuda", generator=g)
    # documents of random length (~ geometric, mean ~3000 tokens), separated by eos
    tokens[torch.rand(bsz, seq_len + 1, device="cuda", generator=g) < 1 / 3000] = Tok.eos_id
    x, y = tokens[:, :-1], tokens[:, 1:]
    torch.cuda.synchronize()
    for k in list(gpu_ms):
        gpu_ms[k] = 0.0
    t0 = time.time()
    loss = model(x, True, targets=y)
    loss = loss[0] if isinstance(loss, tuple) else loss
    loss.float().mean().backward()
    torch.cuda.synchronize()
    dt = (time.time() - t0) * 1000
    model.zero_grad(set_to_none=True)
    print(f"step {step}: loss {loss.float().mean().item():.6f} {dt:.1f} ms (wall, incl. timing syncs); AWM GPU ms: " +
          ", ".join(f"{k}={v:.1f}" for k, v in sorted(gpu_ms.items())))

for name, cs in calls.items():
    print(f"\n{name}:")
    for c in cs[:1]:
        for i, a in enumerate(c):
            print(f"   arg{i}: {a}")
print(f"\npeak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
dist.destroy_process_group()
