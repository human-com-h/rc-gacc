"""Exactly matched node budgets and a separately trainable pair gate."""
import torch
from torch import nn
from .scg_scorer import SCGScorer

class MatchedNodeMLP(nn.Module):
    def __init__(self):
        super().__init__();self.node=nn.Sequential(nn.Linear(768,700),nn.ReLU(),nn.Linear(700,1054),nn.ReLU(),nn.Linear(1054,1))
    def forward(self,h,g):
        logits=self.node(torch.cat([h,g[:,None,:].expand_as(h)],dim=-1)).squeeze(-1)
        return logits,None,torch.zeros((),device=h.device)

class PairGate(nn.Module):
    def __init__(self):
        super().__init__();self.c_head=nn.Sequential(nn.Linear(792,128),nn.ReLU(),nn.Linear(128,4));self.q_head=nn.Sequential(nn.Linear(792,128),nn.ReLU(),nn.Linear(128,5))
    def forward(self,hi,hj,f,is_q):
        cs=torch.zeros_like(f);cs[:,0]=(f[:,0]+f[:,1])/2;cs[:,1]=(f[:,0]-f[:,1]).abs();cs[:,2:5]=f[:,2:5]
        cs[:,5]=(f[:,5]+f[:,6])/2;cs[:,6]=torch.minimum(f[:,5],f[:,6]);cs[:,9:11]=f[:,9:11]
        cs[:,11]=(f[:,11]+f[:,12])/2;cs[:,12]=(f[:,11]-f[:,12]).abs();cs[:,16]=f[:,16]
        cs[:,17:20]=(f[:,17:20]+f[:,20:23])/2;cs[:,20:23]=(f[:,17:20]-f[:,20:23]).abs()
        c=self.c_head(torch.cat([(hi-hj).abs(),hi*hj,cs],dim=-1));c=torch.cat([c,c[:,3:4]],dim=-1)
        q=self.q_head(torch.cat([hi,hj,f],dim=-1));z=torch.where(is_q[:,None],q,c)
        return z[:,:3],z[:,3:]

def base_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k=='W' or k.startswith('node.')}

def base_parameters(model):return sum(v.numel() for k,v in model.named_parameters() if k=='W' or k.startswith('node.'))
