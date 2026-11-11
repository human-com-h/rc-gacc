import numpy as np
from .stage3_decision import decide, render
from .stage4_selection import optimize
from .learned_selector import value

def apply(row,d,parent,u,cov,edges,cal,mode):
    cc=dict(cal,tau_select=float('inf') if cal['tau_select'] is None else cal['tau_select'],tau_bind=float('inf'),tau_bind_learned=cal['tau_bind_learned'])
    if mode=='StrictNoSwap':r=decide(row,d,parent,u,cov,edges,cc)
    else:
        eligible=[e for e in edges if e['syntax_ok'] and e['subtype']!='H_or_unverified' and e['rank_score']>=cc['tau_select']];eligible.sort(key=lambda e:(-e['rank_score'],e['i'],e['j']));chosen=[];used=set()
        for e in eligible:
            if e['i'] in used or e['j'] in used:continue
            z=dict(e,bonus=max(e['effective_deficit'],e['reverse_deficit'] if e['kind']=='C' else 0)*e['reward_keep']+.25*float(np.mean(u[parent['selected']]))*e['pair_keep']);chosen.append(z);used.update([e['i'],e['j']])
            if len(chosen)==2:break
        if chosen:s,active,cert=optimize(d,parent['selected'],u,cov,chosen,parent['budget'])
        else:s,active,cert=parent['selected'],[],dict(status=0,proven_optimal=True,fallback=True,reason='no_eligible_edge',seconds=0.)
        text=render(d,s,[],parent['budget']);bindings=[];used=set();tau=cc['tau_bind_learned']
        for e in sorted(edges,key=lambda e:-e['binding_score']):
            if tau is None or not e['syntax_ok'] or e['subtype']=='H_or_unverified' or e['binding_score']<tau or e['i'] not in s or e['j'] not in s or e['i'] in used or e['j'] in used:continue
            trial=render(d,s,bindings+[e],parent['budget'])
            if trial['within_budget']:bindings.append(e);used.update([e['i'],e['j']]);text=trial
            if len(bindings)==2:break
        r=dict(record_id=row['record_id'],group_id=row['group_id'],split=row['split'],selected=s,reader_text=text['reader_text'],tokens=text['tokens'],budget=parent['budget'],solver=cert,bindings=bindings,labels=len(bindings),protection=active if set(s)!=set(parent['selected']) else [],parent_selected=parent['selected'],base_objective_delta=value(s,u,cov,d['sim'],.15,.8,[])-value(parent['selected'],u,cov,d['sim'],.15,.8,[]))
    S=set(r['selected']);P=set(parent['selected']);r.update(mode=mode,changed=S!=P,removed=sorted(P-S),added=sorted(S-P),source_ids=[d['units'][i]['sid'] for i in r['selected']],source_spans=[[d['units'][i]['raw_start'],d['units'][i]['raw_end']] for i in r['selected']],lexical_coverage=float(d['cov'][:,r['selected']].max(axis=1).sum()) if r['selected'] else 0.)
    return r
