import sys,types,torch
sys.modules.setdefault('flash_attn_2_cuda', types.ModuleType('x'))
import xllm.modules.fused_ops.attention.sca as sca
B,L,H,KV,S,V,C=1,8192,4,4,256,2048,4096
g=torch.Generator(device='cuda').manual_seed(0)
q=torch.randn(B,L,H,S,device='cuda',generator=g).bfloat16(); k=torch.randn(B,L,KV,S,device='cuda',generator=g).bfloat16(); v=torch.randn(B,L,KV,V,device='cuda',generator=g).bfloat16(); dy=torch.randn(B,L,H,V,device='cuda',generator=g).bfloat16()
seg=torch.cumsum((torch.rand(B,L,device='cuda',generator=g)<1/3000).long(),1)
i=torch.arange(L,device='cuda')
mask=(i[None,:]<=i[:,None])&(i[None,:]>=((i[:,None]//C-1).clamp(min=0)*C))&(seg[0][:,None]==seg[0][None,:])
qf,kf,vf=(t.float().requires_grad_(True) for t in (q,k,v))
s=torch.einsum('blhd,bmhd->bhlm',qf,kf)*S**-0.5
p=s.masked_fill(~mask,float('-inf')).softmax(-1)
y=torch.einsum('bhlm,bmhd->blhd',p,vf); y.backward(dy.float())
ref=[qf.grad,kf.grad,vf.grad]
for sp in [1,2,4]:
    sca._SWIFT_QSPLIT=sp
    _,yb,w=sca.sliding_chunk_attention_fwd(q,k,v,C,S**-0.5,None,None,None,seg,0.0,0,'swift',True)
    gr=sca.sliding_chunk_attention_bwd(dy,q,k,v,yb,w,C,S**-0.5,None,None,None,seg,0,False,'swift')[:3]
    print(sp,[f"{((a.float()-b).norm()/b.norm()).item():.2e}/{(a.float()-b).abs().max().item():.2e}" for a,b in zip(gr,ref)])
