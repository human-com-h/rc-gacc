"""Immutable feature readers and deterministic, zero-encoding discourse overlays."""
import hashlib,json,re
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .anchor_satellite import terms

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/phase1_stage3/seed_42'
INDEX=BASE/'recovery_v2/verified_index3497.jsonl'
OUT=ROOT/'results/phase1_final/seed_42'
ANAPH=re.compile(r'^(?:In (?:that|those|this|these)\b|Within (?:that|those|this|the)\b|Under (?:that|those|this|the)\b|For (?:that|those|this|the)\b)',re.I)
SCOPE=re.compile(r'\b(?:if|unless|only|except|provided|under|during|before|after|when|within|whereas|although|historical|revision|account|ethnograph|primary|gazetteer|footprint|earlier|limited|subject to)\b',re.I)
NUM=re.compile(r'\b\d+(?:\.\d+)?\b')
NEG=re.compile(r'\b(?:not|never|cannot|without|no|rejected|fail(?:ed|ure)?)\b',re.I)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def jl(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    tmp.replace(p)

def load(row,verify=False):
    p=ROOT/row['path'].replace('\\', '/');m=json.loads((p/'manifest.json').read_text(encoding='utf8'))
    if sha(p/'manifest.json')!=row['manifest_hash']:raise ValueError('Frozen manifest changed')
    if verify:
        for n,h in m['files'].items():
            if sha(p/n)!=h:raise ValueError('Frozen table changed')
    tables={n:pq.read_table(p/n) for n in m['files']}
    v=tables['vectors.parquet'];g=tables['goal.parquet'].to_pylist()[0];n=len(v)
    matrix=lambda key:np.asarray(v[key].to_pylist(),np.float32)
    return dict(row=row,manifest=m,units=tables['units.parquet'].to_pylist(),h=matrix('embedding'),g=np.asarray(g['goal_vector'],np.float32),
        u=np.asarray(v['utility']),objective_u=np.asarray(v['objective_utility']),cost=np.asarray(v['cost']),
        cov=matrix('coverage').T,sim=matrix('similarity'),lex=matrix('lexical_jaccard'),goal=g,
        edges=tables['edges.parquet'].to_pylist(),budget=tables['budget_features.parquet'].to_pylist())

def complete_claim(u):
    s=u['text'].strip()
    return len(terms(s))>=4 and not s.endswith(('?',':',',',';')) and not re.search(r'^(?:[-*]|\d+[.)])\s|\b(?:def |import |return |if not session|self\.)|[{}=]{2}',s)

def edge_features(d,i,j,kind,raw=None):
    a,b=d['units'][i],d['units'][j];sa,sb=a['text'],b['text'];ta,tb=terms(sa),terms(sb)
    base=d['manifest']['coverage_solutions']['0.1']['selected'];cost=d['cost'];u=d['objective_u'];cov=d['cov']
    cv=cov[:,base].max(axis=1) if base else np.zeros(cov.shape[0]);val=u[base].sum()+.5*cv.sum();rho=val/max(1,cost[base].sum())
    marginal=(0 if j in base else u[j])+.5*(np.maximum(cv,cov[:,j])-cv).sum();raw_deficit=max(0,rho*cost[j]-marginal)
    deficit=0. if j in base else raw_deficit
    nli=(raw or {}).get('nli_forward');rev=(raw or {}).get('nli_reverse')
    f=[d['u'][i],d['u'][j],d['sim'][i,j],len(ta&tb)/max(1,len(ta|tb)),min(abs(a['sid']-b['sid']),20)/20,
       complete_claim(a),complete_claim(b),bool(SCOPE.search(sb)),bool(ANAPH.search(sa)),bool(NEG.search(sa))!=bool(NEG.search(sb)),
       bool(set(NUM.findall(sa))^set(NUM.findall(sb))),i in base,j in base,deficit/max(1,cost[j]),float(marginal),float(rho),
       nli is not None,*((nli or [0,0,0])),*((rev or [0,0,0])),float(kind=='Q')]
    return np.asarray(f,np.float32),dict(claim_in_cov=i in base,qualifier_in_cov=j in base,deficit=float(deficit),raw_price_deficit=float(raw_deficit),marginal=float(marginal),rho=float(rho))

def patch(d,max_q=24,max_c=8):
    units=d['units'];n=len(units);q=[];c=[];tt=[terms(u['text']) for u in units]
    valid=[complete_claim(u) for u in units];ana=[bool(ANAPH.search(u['text'])) for u in units]
    lead=[bool(re.match(r'^(?:if|when|unless|under|provided|subject to)\b',u['text'],re.I)) for u in units]
    scope=[bool(SCOPE.search(u['text'])) for u in units];neg=[bool(NEG.search(u['text'])) for u in units];nums=[set(NUM.findall(u['text'])) for u in units]
    for i,a in enumerate(units):
        if not valid[i]:continue
        for j,b in enumerate(units):
            if i==j:continue
            gap=a['sid']-b['sid'];shared=len(tt[i]&tt[j])
            anaph=ana[i] and gap in [1,2]
            leading=lead[j] and gap in [-2,-1,1,2]
            scoped=scope[j] and 0<abs(gap)<=2 and shared>=2
            if anaph or leading or scoped:
                score=20*anaph+5*leading+2*scoped+shared/max(1,len(tt[i]|tt[j]))+float(d['u'][i])+float(d['sim'][i,j])
                q.append((score,i,j))
            if i<j and shared>=3 and float(d['sim'][i,j])>=.55:
                opposite=neg[i]!=neg[j];different=bool(nums[i] and nums[j] and nums[i]!=nums[j])
                if opposite or different:c.append((3*opposite+different+float(d['sim'][i,j])+shared/50,i,j))
    out=[]
    for kind,proposals,limit in [('Q',q,max_q),('C',c,max_c)]:
        for score,i,j in sorted(proposals,key=lambda x:(-x[0],x[1],x[2]))[:limit]:
            raw=next((e for e in d['edges'] if e['i']==i and e['j']==j),None)
            f,b=edge_features(d,i,j,kind,raw)
            out.append(dict(i=i,j=j,kind=kind,source_ids=[units[i]['sid'],units[j]['sid']],score=score,features=f.tolist(),
                budget10=b,nli_available=bool(raw and raw['nli_available']),origin='cached_discourse_patch_v1',semantic_verified=False))
    return out

def view_edges(d,overlay):
    result={}
    for e in d['edges']:
        kind=e['prior_kind'] if e['prior_kind'] in ['C','Q'] else ('Q' if SCOPE.search(d['units'][e['j']]['text']) else 'C')
        f,b=edge_features(d,e['i'],e['j'],kind,e)
        result[(e['i'],e['j'],kind)]=dict(i=e['i'],j=e['j'],kind=kind,features=f.tolist(),budget10=b,origin='frozen_base')
    for e in overlay:result[(e['i'],e['j'],e['kind'])]=e
    return sorted(result.values(),key=lambda e:(e['kind'],e['i'],e['j']))
