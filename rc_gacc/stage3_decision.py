import copy, re, time
import numpy as np
import torch
import tiktoken
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from scipy.special import expit
from .stage2_selection import candidates
from .stage3_views import proposals, claim_ok, qualifier_ok
from .learned_selector import value

def source_certificate(d,e):
    """Restricted Q syntax witness, independent of training or oracle labels."""
    if e['kind']!='Q':return False
    a,b=d['units'][e['i']],d['units'][e['j']];text=b['text'].strip()
    adjacent_before=0<=a['raw_start']-b['raw_end']<=4
    dependent=bool(re.match(r'^(?:If|Unless|Under|When|Provided|Subject to)\b',text,re.I)) and text.endswith(',') and not a['text'].lstrip().startswith('Document [')
    adjacent_after=0<=b['raw_start']-a['raw_end']<=4
    explicit_pointer=bool(re.match(r'^The preceding (?:claim|assertion)\b.*\bonly (?:if|when)\b',text,re.I))
    return bool(e['syntax_ok'] and ((adjacent_before and dependent) or (adjacent_after and explicit_pointer)))

def scored(gate,d,overlay,parent,u,cov,cal):
    pp=proposals(d,overlay);temp=copy.deepcopy(d);temp['edges']=[]
    es=candidates(temp,pp,parent,u,cov)
    if not es:return []
    ii=[e['i'] for e in es];jj=[e['j'] for e in es]
    with torch.no_grad():z,k=gate(torch.tensor(d['h'][ii]),torch.tensor(d['h'][jj]),torch.tensor([e['features'] for e in es],dtype=torch.float32),torch.tensor([e['kind']=='Q' for e in es]))
    zz=z.numpy();kk=k.numpy();pr=expit(zz/np.asarray(cal['temperatures'])+np.asarray(cal['biases']));pk=expit(kk/float(cal.get('keep_T',1))+float(cal.get('keep_bias',0)))
    for n,e in enumerate(es):
        e['raw_logits']=zz[n].tolist();e['raw_keep_logits']=kk[n].tolist();e['probabilities']=pr[n].tolist();e['p_keep']=pk[n].tolist();e['pair_keep']=float(np.sqrt(pk[n].prod()))
        e['reward_keep']=float(e['pair_keep']*(.5+.5*pr[n,2]));e['rank_score']=float(pr[n,0]*pr[n,1]*e['reward_keep']);e['binding_score']=float(pr[n,0]*pr[n,1]*(.5+.5*e['pair_keep']))
        e['syntax_ok']=claim_ok(d['units'][e['i']]) and (qualifier_ok(d['units'][e['j']]) if e['kind']=='Q' else claim_ok(d['units'][e['j']]))
        e['source_syntax_certificate']=source_certificate(d,e)
    return es

def render(d,selected,bindings,budget):
    tok=tiktoken.get_encoding('cl100k_base');s=set(selected);used=set();starts={};accepted=[]
    for e in bindings:
        i,j=e['i'],e['j']
        if i not in s or j not in s or i in used or j in used:continue
        a,b=d['units'][i]['text'],d['units'][j]['text'];anaph=bool(re.match(r'^(?:The preceding|This |That |These |Those |It |They )',b,re.I))
        order=[i,j] if e['kind']=='C' or anaph else [j,i]
        starts[min(i,j)]=(e,order);used.update([i,j]);accepted.append(e)
    emitted=set();parts=[]
    for i in sorted(s):
        if i in starts:
            e,order=starts[i];parts.extend([e['kind']+': '+d['units'][order[0]]['text'],d['units'][order[1]]['text']]);emitted.update(order)
        elif i not in emitted and i not in used:parts.append(d['units'][i]['text'])
    text='\n'.join(parts);tokens=len(tok.encode(text,disallowed_special=()))
    return dict(reader_text=text,tokens=tokens,budget=budget,labels=len(accepted),bindings=accepted,within_budget=tokens<=budget)

def additive_select(d,parent,u,cov,edges,budget):
    """Binary choices over <=12 edges; every original parent unit is fixed."""
    tok=tiktoken.get_encoding('cl100k_base');P=set(parent);plain=render(d,parent,[],budget);candidates_e=[]
    for e in edges:
        missing=set([e['i'],e['j']])-P
        if missing:candidates_e.append(dict(e,missing=sorted(missing)))
    candidates_e=candidates_e[:12]
    if not candidates_e:return parent,[],dict(status=0,proven_optimal=True,reason='no_additive_options',seconds=0.)
    nodes=sorted({i for e in candidates_e for i in e['missing']});idx={i:n for n,i in enumerate(nodes)};m=len(nodes);q=len(candidates_e);cv=cov[:,sorted(P)].max(axis=1) if P else np.zeros(len(cov));delta=np.maximum(0,cov[:,nodes]-cv[:,None]);k=len(delta)
    # Fixed-parent exact text cost is a constant; extra chunks are conservative.
    costs=np.array([len(tok.encode('\n'+d['units'][i]['text'],disallowed_special=()))+2 for i in nodes],float);residual=budget-plain['tokens']
    redund=[(a,b,max(0.,float(d['sim'][i,j])-.8)) for a,i in enumerate(nodes) for b,j in enumerate(nodes) if a<b and d['sim'][i,j]>.8]
    y0=m;v0=m+q;z0=v0+k*m;size=z0+len(redund);objective=np.zeros(size)
    objective[:m]=np.array([-float(u[i]-.15*sum(max(0,float(d['sim'][i,j])-.8) for j in P)) for i in nodes])
    objective[v0:z0]=-.5*delta.ravel()
    scale=float(np.mean([u[i] for i in P])) if P else float(np.mean(u))
    for t,e in enumerate(candidates_e):objective[y0+t]=-(max(e['effective_deficit'],e.get('reverse_deficit',0) if e['kind']=='C' else 0)*e['reward_keep']+.25*scale*e['pair_keep'])
    for t,(_,_,w) in enumerate(redund):objective[z0+t]=.15*w
    rr=[];cc=[];vv=[];lb=[];ub=[]
    def add(vals,hi,lo=-np.inf):
        n=len(lb)
        for col,val in vals.items():
            if val:rr.append(n);cc.append(col);vv.append(val)
        lb.append(lo);ub.append(hi)
    add({i:costs[i] for i in range(m)},residual);add({y0+t:1 for t in range(q)},2)
    for t,e in enumerate(candidates_e):
        for i in e['missing']:add({y0+t:1,idx[i]:-1},0)
    for i in nodes:add({idx[i]:1,**{y0+t:-1 for t,e in enumerate(candidates_e) if i in e['missing']}},0)
    for i in {n for e in candidates_e for n in [e['i'],e['j']]}:add({y0+t:1 for t,e in enumerate(candidates_e) if i in [e['i'],e['j']]},1)
    for a in range(k):
        add({v0+a*m+n:1 for n in range(m)},1)
        for n in range(m):add({v0+a*m+n:1,n:-1},0)
    for t,(a,b,_) in enumerate(redund):
        z=z0+t;add({z:1,a:-1},0);add({z:1,b:-1},0);add({a:1,b:1,z:-1},1)
    # Linear lower bound on the exact base-value increment, excluding rewards.
    # Max-coverage assignment can only understate the true coverage increment.
    base_row={i:-float(objective[i]) for i in range(m)}
    base_row.update({v0+n:.5*float(v) for n,v in enumerate(delta.ravel())})
    base_row.update({z0+t:-.15*w for t,(_,_,w) in enumerate(redund)})
    add(base_row,np.inf,lo=0.)
    integ=np.zeros(size);integ[:m+q]=1;matrix=coo_matrix((vv,(rr,cc)),shape=(len(lb),size)).tocsc();start=time.perf_counter()
    result=milp(objective,integrality=integ,bounds=Bounds(0,1),constraints=LinearConstraint(matrix,lb,ub),options={'time_limit':1.,'mip_rel_gap':0.})
    if result.x is None:return parent,[],dict(status=int(result.status),proven_optimal=False,reason='no_incumbent_keep_parent',seconds=time.perf_counter()-start)
    chosen=[e for t,e in enumerate(candidates_e) if result.x[y0+t]>.5];selected=sorted(P|{i for n,i in enumerate(nodes) if result.x[n]>.5});actual=render(d,selected,[],budget)
    f0=value(parent,u,cov,d['sim'],.15,.8,[]);f1=value(selected,u,cov,d['sim'],.15,.8,[])
    if not P.issubset(selected) or not actual['within_budget'] or f1<f0-1e-6:
        return parent,[],dict(status=int(result.status),proven_optimal=False,proposal_proven_optimal=bool(result.status==0),reason='parent_safety_guard',proposed_base_delta=f1-f0,seconds=time.perf_counter()-start)
    assert len(chosen)<=2 and all(e['i'] in selected and e['j'] in selected for e in chosen)
    return selected,chosen,dict(status=int(result.status),proven_optimal=bool(result.status==0),mip_gap=getattr(result,'mip_gap',None),base_delta=f1-f0,seconds=time.perf_counter()-start)

def decide(row,d,parent,u,cov,edges,cal):
    budget=parent['budget'];P=set(parent['selected']);eligible=[e for e in edges if e['syntax_ok'] and e['subtype']!='H_or_unverified' and e['rank_score']>=cal['tau_select']]
    eligible.sort(key=lambda e:(-e['rank_score'],e['i'],e['j']))
    selected,protection,cert=additive_select(d,parent['selected'],u,cov,eligible,budget)
    # Binding needs no extra protection bonus and cannot evict a single unit.
    bindings=[];used=set();rendered=render(d,selected,[],budget);rejected=[]
    learned_tau=cal.get('tau_bind_learned',cal['tau_bind'])
    available=[e for e in edges if e['syntax_ok'] and e['subtype']!='H_or_unverified' and e['i'] in selected and e['j'] in selected and
        ((learned_tau is not None and e['binding_score']>=learned_tau) or (e['source_syntax_certificate'] and e['probabilities'][1]>=.25))]
    for e in sorted(available,key=lambda e:(-e['binding_score'],e['i'],e['j'])):
        if len(bindings)>=2:break
        if e['i'] in used or e['j'] in used:continue
        trial=render(d,selected,bindings+[e],budget)
        if trial['within_budget']:bindings.append(e);used.update([e['i'],e['j']]);rendered=trial
        else:rejected.append(dict(i=e['i'],j=e['j'],reason='binding_token_budget'))
    assert P.issubset(selected) and rendered['within_budget']
    unresolved=[dict(i=e['i'],j=e['j'],kind=e['kind'],reason='not_closed_under_parent_containment_budget_guard') for e in eligible if ((e['i'] in selected)!=(e['j'] in selected) if e['kind']=='C' else (e['i'] in selected and e['j'] not in selected))]
    return dict(record_id=row['record_id'],group_id=row['group_id'],split=row['split'],variant='+CleanKeepSafe',selected=selected,source_ids=[d['units'][i]['sid'] for i in selected],
        source_spans=[[d['units'][i]['raw_start'],d['units'][i]['raw_end']] for i in selected],reader_text=rendered['reader_text'],tokens=rendered['tokens'],budget=budget,labels=rendered['labels'],bindings=bindings,protection=protection,
        solver=cert,parent_selected=parent['selected'],parent_tokens=parent['tokens'],parent_containment=True,base_objective_delta=value(selected,u,cov,d['sim'],.15,.8,[])-value(parent['selected'],u,cov,d['sim'],.15,.8,[]),
        lexical_coverage=float(d['cov'][:,selected].max(axis=1).sum()) if selected else 0.,binding_only=selected==parent['selected'] and bool(bindings),added_units=len(set(selected)-P),unresolved_risks=unresolved,binding_rejected=rejected,submitted=False)
