import sys,types,torch
sys.modules.setdefault('flash_attn_2_cuda', types.ModuleType('x'))
from torch.profiler import profile, ProfilerActivity
from xllm.modules.fused_ops.conv import causal_conv1d as cc
from xllm.modules.fused_ops.cute.causal_conv import conv_fwd, conv_bwd
def t(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(10): fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in p.key_averages())/10
for B,L,D,act in [(2,16384,2048,None),(2,16384,2048,'silu'),(1,8192,8192,'silu'),(1,8192,2048,None)]:
    g=torch.Generator(device='cuda').manual_seed(0)
    x=torch.randn(B,L,D,device='cuda',generator=g).bfloat16(); w=torch.randn(D,4,device='cuda',generator=g).bfloat16()
    bos=torch.rand(B,L,device='cuda',generator=g)<1/3000; dy=torch.randn_like(x)
    tf=t(lambda: cc._triton_causal_conv1d_fwd(x,w,None,None,bos,False,act,False))
    cf=t(lambda: conv_fwd(x,w,bos,act))
    tb=t(lambda: cc._triton_causal_conv1d_bwd(dy,None,x,w,None,None,bos,act,False))
    cb=t(lambda: conv_bwd(dy,x,w,bos,act))
    y1=cc._triton_causal_conv1d_fwd(x,w,None,None,bos,False,act,False)[0]; y2=conv_fwd(x,w,bos,act)
    print(f"B{B} L{L} D{D} {act}: fwd triton {tf:6.0f} cute {cf:6.0f} us | bwd triton {tb:6.0f} cute {cb:6.0f} us | max|dy| {(y1.float()-y2.float()).abs().max().item():.2e}")
