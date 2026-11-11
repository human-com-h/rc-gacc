"""Trainable 1M+ scorer. This module has no encoder, NLI, oracle or network calls."""
import torch
from torch import nn
from torch.nn import functional as F

class SCGScorer(nn.Module):
    def __init__(self,dim=384,k=8,edge_dim=24):
        super().__init__();self.k=k
        self.W=nn.Parameter(torch.empty(k,dim,dim));nn.init.normal_(self.W,std=.01)
        with torch.no_grad():self.W.add_(torch.eye(dim).unsqueeze(0)*.1)
        self.node=nn.Sequential(nn.Linear(2*dim,128),nn.ReLU(),nn.Linear(128,1))
        self.c_head=nn.Sequential(nn.Linear(2*dim+edge_dim,128),nn.ReLU(),nn.Linear(128,3))
        self.q_head=nn.Sequential(nn.Linear(2*dim+edge_dim,128),nn.ReLU(),nn.Linear(128,3))
    def forward(self,h,g):
        probes=F.normalize(torch.einsum('bd,kde->bke',g,self.W),dim=-1)
        logits=self.node(torch.cat([h,g[:,None,:].expand_as(h)],dim=-1)).squeeze(-1)
        coverage=torch.einsum('bnd,bkd->bkn',h,probes).clamp(min=0)/self.k
        gram=probes@probes.transpose(-1,-2);eye=torch.eye(self.k,device=h.device)
        orthogonal=((gram-eye)**2).mean()
        return logits,coverage,orthogonal
    def relation(self,hi,hj,features,is_q):
        # Every C feature retained here is invariant under endpoint exchange.
        f=features;cs=torch.zeros_like(f)
        cs[:,0]=(f[:,0]+f[:,1])/2;cs[:,1]=(f[:,0]-f[:,1]).abs();cs[:,2:5]=f[:,2:5]
        cs[:,5]=(f[:,5]+f[:,6])/2;cs[:,6]=torch.minimum(f[:,5],f[:,6]);cs[:,9:11]=f[:,9:11]
        cs[:,11]=(f[:,11]+f[:,12])/2;cs[:,12]=(f[:,11]-f[:,12]).abs();cs[:,16]=f[:,16]
        cs[:,17:20]=(f[:,17:20]+f[:,20:23])/2;cs[:,20:23]=(f[:,17:20]-f[:,20:23]).abs()
        c=self.c_head(torch.cat([(hi-hj).abs(),hi*hj,cs],dim=-1))
        q=self.q_head(torch.cat([hi,hj,f],dim=-1))
        return torch.where(is_q[:,None],q,c)

def fixed_teacher_anticollapse(node_prob_i,node_prob_j,teacher_mask):
    # The teacher mask is a detached label. Predicted gate scores cannot turn it off.
    loss=(node_prob_i-node_prob_j).abs()+F.relu(.6-torch.minimum(node_prob_i,node_prob_j))
    return (loss*teacher_mask).sum()/teacher_mask.sum().clamp(min=1)
