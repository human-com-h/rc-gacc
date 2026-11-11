"""Sparse exact binary selector with explicit redundancy AND variables."""
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from time import perf_counter

def value(selected,u,cov,sim,beta,tau,relations):
    s=set(selected);v=sum(float(u[i]) for i in s)+(.5*np.max(cov[:,sorted(s)],axis=1).sum() if s and cov.size else 0)
    v-=beta*sum(max(0,float(sim[i,j])-tau) for i in s for j in s if i<j)
    v+=sum(e.get('bonus',0) for e in relations if e['i'] in s and e['j'] in s)
    return float(v)

def select(cost,budget,u,cov,sim,beta=.15,tau=.8,relations=(),time_limit=1.):
    cost=np.asarray(cost);u=np.asarray(u);cov=np.asarray(cov);n=len(u);k=len(cov)
    redundant=[(i,j,float(sim[i,j])-tau) for i in range(n) for j in range(i+1,n) if beta>0 and sim[i,j]>tau]
    y0=n+k*n;z0=y0+len(redundant);size=z0+len(relations);c=np.zeros(size);c[:n]=-u;c[n:y0]=-.5*cov.ravel()
    for t,(_,_,w) in enumerate(redundant):c[y0+t]=beta*w
    for t,e in enumerate(relations):c[z0+t]=-e.get('bonus',0)
    rr=[];cc=[];vv=[];lb=[];ub=[]
    def add(vals,lo=-np.inf,hi=np.inf):
        idx=len(lb)
        for col,v in vals.items():
            if v:rr.append(idx);cc.append(col);vv.append(v)
        lb.append(lo);ub.append(hi)
    b={i:float(cost[i]) for i in range(n)};b.update({z0+t:float(e.get('label_cost',0)) for t,e in enumerate(relations)})
    add(b,hi=budget)
    for z in range(k):
        add({n+z*n+i:1 for i in range(n)},hi=1)
        for i in range(n):add({n+z*n+i:1,i:-1},hi=0)
    def and_pair(i,j,z):
        add({z:1,i:-1},hi=0);add({z:1,j:-1},hi=0);add({i:1,j:1,z:-1},hi=1)
    for t,(i,j,_) in enumerate(redundant):and_pair(i,j,y0+t)
    for t,e in enumerate(relations):
        i,j=e['i'],e['j'];and_pair(i,j,z0+t)
        if e['kind']=='Q':add({i:1,j:-1},hi=0)
        elif e['kind']=='C':add({i:1,j:-1},lo=0,hi=0)
        else:raise ValueError('Only direct C/Q may impose closure; H has no pair lock')
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(lb),size)).tocsc();integrality=np.zeros(size);integrality[:n]=1
    started=perf_counter();r=milp(c,integrality=integrality,bounds=Bounds(0,1),constraints=LinearConstraint(matrix,lb,ub),options={'time_limit':time_limit,'mip_rel_gap':0.0})
    selected=np.flatnonzero(r.x[:n]>.5).tolist() if r.x is not None else [];s=set(selected)
    spent=int(cost[selected].sum())+sum(e.get('label_cost',0) for e in relations if e['i'] in s and e['j'] in s)
    violations=sum((e['i'] in s and e['j'] not in s) if e['kind']=='Q' else ((e['i'] in s)!=(e['j'] in s)) for e in relations)
    if spent>budget or violations:raise RuntimeError('Independent MILP feasibility check failed')
    objective=value(selected,u,cov,sim,beta,tau,relations)
    if r.fun is not None and abs(objective+r.fun)>1e-5*(1+abs(objective)):raise RuntimeError('Independent objective mismatch')
    return selected,dict(status=int(r.status),proven_optimal=bool(r.status==0 and r.x is not None),mip_gap=getattr(r,'mip_gap',None),
        objective=objective,selected_proxy_cost=spent,hard_violations=violations,redundancy_pairs=len(redundant),seconds=perf_counter()-started)
