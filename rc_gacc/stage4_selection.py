"""Observable-floor controlled swap, with exact independent checks."""
import time,math,tiktoken
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from .stage3_decision import render
from .learned_selector import value

def removable(parent,u,cov,sim):
    p=set(parent);f=value(parent,u,cov,sim,.15,.8,[]);m={i:f-value(sorted(p-{i}),u,cov,sim,.15,.8,[]) for i in p};low=set(sorted(p,key=lambda i:(m[i],i))[:max(1,math.ceil(.4*len(p)))])
    return sorted(low|{i for i in p if any(j!=i and sim[i,j]>=.85 for j in p)})

def optimize(d,parent,u,cov,edges,budget,force_pair=False,strict=False,hard_q=()):
    P=set(parent);es=edges[:12];U=sorted(P|{i for e in es for i in [e['i'],e['j']]}|{i for pair in hard_q for i in pair});ix={i:n for n,i in enumerate(U)};n=len(U);k=len(cov);q=len(es);allow=set() if strict else set(removable(parent,u,cov,d['sim']));red=[(a,b,max(0,float(d['sim'][i,j])-.8)) for a,i in enumerate(U) for b,j in enumerate(U) if a<b and d['sim'][i,j]>.8]
    v0=n;z0=n+k*n;y0=z0+len(red);size=y0+q;c=np.zeros(size);c[:n]=-u[U];c[v0:z0]=-.5*cov[:,U].ravel()
    for t,(_,_,w) in enumerate(red):c[z0+t]=.15*w
    for t,e in enumerate(es):c[y0+t]=-e.get('bonus',0.)
    lo=np.zeros(size);hi=np.ones(size)
    for i in P-allow:lo[ix[i]]=1
    if force_pair:
        for e in es:lo[ix[e['i']]]=lo[ix[e['j']]]=1
    rr=[];cc=[];vv=[];lb=[];ub=[]
    def add(vals,lower=-np.inf,upper=np.inf):
        row=len(lb)
        for col,val in vals.items():
            if val:rr.append(row);cc.append(col);vv.append(val)
        lb.append(lower);ub.append(upper)
    tok=tiktoken.get_encoding('cl100k_base');costs={a:len(tok.encode(d['units'][i]['text']+'\n',disallowed_special=())) for a,i in enumerate(U)}
    add(costs,upper=budget);add({ix[i]:-1 for i in allow},upper=6-len(allow))
    for claim,qualifier in hard_q:
        add({ix[claim]:1,ix[qualifier]:-1},upper=0)
    pc=cov[:,sorted(P)].max(axis=1) if P else np.zeros(k)
    for dim in range(k):
        add({v0+dim*n+a:1 for a in range(n)},upper=1)
        for a in range(n):add({v0+dim*n+a:1,a:-1},upper=0)
        add({v0+dim*n+a:float(cov[dim,i]) for a,i in enumerate(U)},lower=.95*pc[dim])
    def and_pair(a,b,z):
        add({z:1,a:-1},upper=0);add({z:1,b:-1},upper=0);add({a:1,b:1,z:-1},upper=1)
    for t,(a,b,_) in enumerate(red):and_pair(a,b,z0+t)
    for t,e in enumerate(es):and_pair(ix[e['i']],ix[e['j']],y0+t)
    # Exclude relation rewards from the frozen base-value floor.
    add({**{a:float(u[i]) for a,i in enumerate(U)},**{v0+a:.5*float(v) for a,v in enumerate(cov[:,U].ravel())},**{z0+t:-.15*w for t,(_,_,w) in enumerate(red)}},lower=(1. if strict else .98)*value(parent,u,cov,d['sim'],.15,.8,[]))
    integ=np.zeros(size);integ[:n]=1;start=time.perf_counter();r=milp(c,integrality=integ,bounds=Bounds(lo,hi),constraints=LinearConstraint(coo_matrix((vv,(rr,cc)),shape=(len(lb),size)).tocsc(),lb,ub),options={'time_limit':1.,'mip_rel_gap':0.})
    cert=dict(status=int(r.status),proven_optimal=bool(r.status==0 and r.x is not None),proven_infeasible=bool(r.status==2),mip_gap=getattr(r,'mip_gap',None),seconds=time.perf_counter()-start,scope='Restricted parent+edge-endpoint search with observable floors; no global semantic guarantee')
    if r.x is None:return list(parent),[],dict(cert,fallback=True,reason='no_feasible_incumbent')
    selected=[i for a,i in enumerate(U) if r.x[a]>.5];S=set(selected);out=render(d,selected,[],budget);f0=value(parent,u,cov,d['sim'],.15,.8,[]);f1=value(selected,u,cov,d['sim'],.15,.8,[]);cv=cov[:,selected].max(axis=1) if selected else np.zeros(k)
    valid=out['within_budget'] and len(P-S)<=6 and (P-allow).issubset(S) and f1+1e-6>=(1 if strict else .98)*f0 and np.all(cv+1e-6>=.95*pc)
    if force_pair:valid=valid and all(e['i'] in S and e['j'] in S for e in es)
    valid=valid and all(claim not in S or qualifier in S for claim,qualifier in hard_q)
    if not valid:return list(parent),[],dict(cert,proven_optimal=False,proposal_proven_optimal=cert['proven_optimal'],fallback=True,reason='exact_output_guard')
    active=[e for e in es if e['i'] in S and e['j'] in S];return selected,active,dict(cert,fallback=False,removed=sorted(P-S),base_ratio=f1/max(1e-12,f0),minimum_subgoal_ratio=float(np.min(cv[pc>0]/pc[pc>0])) if np.any(pc>0) else 1.)
