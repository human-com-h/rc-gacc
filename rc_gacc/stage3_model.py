"""No weak-label gradients enter type or keep: separate tiny relevance branch."""
import torch
from torch import nn
from .stage2_models import PairGate

class CleanGate(nn.Module):
    def __init__(self):
        super().__init__();self.pair=PairGate();self.weak_goal=nn.Sequential(nn.Linear(4,16),nn.ReLU(),nn.Linear(16,1))
    def weak_logits(self,f):
        x=torch.stack([torch.maximum(f[:,0],f[:,1]),torch.minimum(f[:,0],f[:,1]),f[:,2],f[:,3]],dim=-1).detach()
        return self.weak_goal(x).squeeze(-1)
    def forward(self,hi,hj,f,is_q):
        z,k=self.pair(hi,hj,f,is_q);goal=.75*z[:,1]+.25*self.weak_logits(f);z=torch.stack([z[:,0],goal,z[:,2]],dim=-1)
        return z,k

class GoalConditionedGate(CleanGate):
    """Explicit cached goal and frozen Wk responses; old CleanGate unchanged."""
    def __init__(self):
        super().__init__();self.context=nn.Sequential(nn.Linear(1168,96),nn.ReLU(),nn.Linear(96,5));self.direct_goal=nn.Sequential(nn.Linear(1168,64),nn.ReLU(),nn.Linear(64,1))
    def contextual(self,hi,hj,g,response,is_q):
        q=is_q[:,None];a=torch.where(q,hi,(hi+hj)/2)*g;b=torch.where(q,hj,(hi-hj).abs())*g
        r1,r2=response[:,:8],response[:,8:];r=torch.cat([torch.where(q,r1,(r1+r2)/2),torch.where(q,r2,(r1-r2).abs())],dim=-1)
        return torch.cat([g,a,b,r],dim=-1)
    def weak_context_logits(self,hi,hj,g,response,is_q):
        return self.direct_goal(self.contextual(hi,hj,g,response,is_q).detach()).squeeze(-1)
    def forward(self,hi,hj,f,is_q,g,response):
        ctx=self.contextual(hi,hj,g,response,is_q);z,k=self.pair(hi,hj,f,is_q);delta=self.context(ctx);z=z+.25*delta[:,:3]
        z=torch.stack([z[:,0],.75*z[:,1]+.25*self.direct_goal(ctx).squeeze(-1),z[:,2]],dim=-1);k=k+.25*delta[:,3:]
        mean=k.mean(dim=1,keepdim=True);k=torch.where(is_q[:,None],k,mean.expand_as(k));return z,k
