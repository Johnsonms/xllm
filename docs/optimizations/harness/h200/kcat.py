"""Kernel time per step by category (FWDBWD window), to compare with the H100 per-kernel table. usage: kcat.py trace.json"""
import json, re, sys
from collections import defaultdict
tr = json.load(open(sys.argv[1]))["traceEvents"]
win = [e for e in tr if e.get("cat") == "user_annotation" and e["name"] == "FWDBWD"][0]; w0, w1 = win["ts"], win["ts"] + win["dur"]
CATS = [("TSDN", r"timestep_decay_norm"), ("RoPE / q-k prep", r"qk_prep(?!.*Conv)|rope"), ("conv + q-k prep", r"conv_qk_prep|ConvQk"),
        ("v conv", r"causal_conv|ConvAct"), ("attn gate", r"attn_gate|AttnGate|_gate_kernel"), ("softmax", r"softmax"),
        ("layer / group norm", r"layer_norm|LnFwd|LnBwd|group_norm"), ("cross entropy rows", r"cross_entropy|CrossEntropy"),
        ("SwiGLU", r"swiglu|SwiGLU|_swiglu"), ("AWM", r"adaptive_working_memory|Awm"), ("fp8 quantise", r"_fp8_kernel|_transpose_kernel"),
        ("xattn fwd", r"AttentionFwdSm90"), ("xattn bwd (all)", r"AttentionBwd|AttentionCast|xattn"),
        ("GEMM (cuBLAS / quack / DeepGEMM)", r"nvjet|gemm|Gemm|cutlass|sm90_xmma|deep_gemm"),
        ("copies", r"[Cc]opy|Memcpy|cat_|CatArray|split_with_sizes"), ("optimizer / foreach", r"multi_tensor|adam|Adam|foreach")]
agg = defaultdict(float); tot = 0
for e in tr:
    if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and w0 <= e["ts"] <= w1 and "nccl" not in e["name"].lower():
        d = e["dur"] / 1000; tot += d
        for c, p in CATS:
            if re.search(p, e["name"]): agg[c] += d; break
        else: agg["other"] += d
print(f"window {win['dur']/1000:.0f} ms, kernel time {tot:.0f} ms")
for c, _ in CATS + [("other", "")]:
    print(f"  {c:34s} {agg[c]:7.1f} ms")
