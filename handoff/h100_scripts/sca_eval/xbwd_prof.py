import sys,torch
from torch.profiler import profile, ProfilerActivity
from xattn.ops import flash_sca_fwd, flash_sca_bwd
B,H,KV,S,V=2,8,4,256,256; L,C=16384,4096; mode=sys.argv[1]
g=torch.Generator(device="cuda").manual_seed(0)
q=torch.randn(B,L,H,S,device="cuda",generator=g).bfloat16(); k=torch.randn(B,L,KV,S,device="cuda",generator=g).bfloat16()
v=torch.randn(B,L,KV,V,device="cuda",generator=g).bfloat16(); dy=torch.randn(B,L,H,V,device="cuda",generator=g).bfloat16()
bos=torch.rand(B,L,device="cuda",generator=g)<(1/3000 if mode=="eos" else 0); bos[:,0]=True
y,_,lse=flash_sca_fwd(q,k,v,C,S**-0.5,None,None,bos_mask=bos)
f=lambda: flash_sca_bwd(dy,q,k,v,y,lse,C,S**-0.5,None,None,bos_mask=bos)
for _ in range(3): f()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(5): f()
    torch.cuda.synchronize()
for e in sorted([e for e in p.key_averages() if e.device_time_total>0], key=lambda e:-e.device_time_total)[:10]: print(f'{e.device_time_total/5:8.1f} us {e.count//5}x {e.key[:1200]}')
