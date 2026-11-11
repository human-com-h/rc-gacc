import copy
import numpy as np
import torch
from .frozen_views import view_edges
from .learned_selector import value

def utility(m,d):
    with torch.no_grad():z,c,_=m(torch.tensor(d['h'])[None],torch.tensor(d['g'])[None]);p=torch.sigmoid(z)[0].numpy()
    cov=d['cov'] if c is None else np.concatenate([.5*d['cov'],.5*c[0].numpy()],axis=0)
    return d['objective_u']*(.5+p),cov

def candidates(d,overlay,parent,u,cov):
    es=copy.deepcopy(view_edges(d,overlay));s=set(parent);rho=value(parent,u,cov,d['sim'],.15,.8,[])/max(1,sum(d['cost'][i] for i in s));cv=cov[:,sorted(s)].max(axis=1) if s else np.zeros(len(cov))
    for e in es:
        i,j=e['i'],e['j'];marginal=0. if j in s else float(u[j]+.5*(np.maximum(cv,cov[:,j])-cv).sum()-.15*sum(max(0,float(d['sim'][j,k])-.8) for k in s))
        deficit=0. if j in s else max(0.,float(rho*(d['cost'][j]+3)-marginal));f=e['features'];f[11]=float(i in s);f[12]=float(j in s);f[13]=deficit/max(1,d['cost'][j]);f[14]=marginal;f[15]=rho;e['effective_deficit']=deficit
        reverse_marginal=0. if i in s else float(u[i]+.5*(np.maximum(cv,cov[:,i])-cv).sum()-.15*sum(max(0,float(d['sim'][i,k])-.8) for k in s))
        e['reverse_deficit']=0. if i in s else max(0.,float(rho*(d['cost'][i]+3)-reverse_marginal))
    return es
