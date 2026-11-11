"""Zero-encoding subtype/syntax views; dependent qualifiers are not bad claims."""
import re,copy
import numpy as np
from .frozen_views import edge_features,view_edges,complete_claim
from .anchor_satellite import terms

COND=re.compile(r'\b(?:if|unless|only when|only if|subject to|under|when|depends? on|contingent|except)\b|^Provided\b|\bprovided (?:that|they|it|you|we|the|a|there)\b',re.I)
SCOPE=re.compile(r'\b(?:gazetteer|archive|census|register|record|scope|historical|rather than|agency|issuing|recruiting|not equivalent|definition|convention)\w*\b',re.I)
META=re.compile(r'^In (?:this|our) (?:paper|work|review|study|section)\b',re.I)

def claim_ok(unit):
    t=re.sub(r'^\s*[-*•]\s*','',unit['text']).strip()
    if not t or t.endswith(('?',':')) or re.search(r'^(?:def |class |return |import |\[start|\[end|Document \[\d+\]Adapted)',t):return False
    if re.fullmatch(r'[\W\d_]+',t):return False
    if len(re.findall(r'[\u4e00-\u9fff]',t))>=12:return t.endswith('。') and bool(re.search(r'是|系|应|认为|认定|不得|可以|履行|支付|已|有权',t))
    return len(terms(t))>=4 and not t.endswith((',', ';'))

def qualifier_ok(unit):
    t=re.sub(r'^\s*[-*•]\s*','',unit['text']).strip()
    if re.match(r'^(?:If|Unless|Under|When|Provided|Subject to)\b',t,re.I):return len(terms(t))>=2 and len(t)>8
    return claim_ok(dict(text=t))

def q_subtype(a,b):
    if COND.search(b):return 'Q_cond'
    if SCOPE.search(a+' '+b):return 'Q_scope'
    return 'H_or_unverified'

def proposals(d,overlay):
    out={}
    def add(i,j,kind,raw=None,origin='derived',observed_features=None):
        if i==j:return
        f,b=edge_features(d,i,j,kind,raw);f[5]=float(claim_ok(d['units'][i]));f[6]=float(qualifier_ok(d['units'][j]) if kind=='Q' else claim_ok(d['units'][j]))
        if raw is None and observed_features is not None and observed_features[16]>0:f[16:23]=observed_features[16:23]
        out[(i,j,kind)]=dict(i=i,j=j,kind=kind,features=f.tolist(),budget10=b,origin=origin,subtype=q_subtype(d['units'][i]['text'],d['units'][j]['text']) if kind=='Q' else 'C')
    for e in view_edges(d,overlay):add(e['i'],e['j'],e['kind'],next((r for r in d['edges'] if r['i']==e['i'] and r['j']==e['j']),None),e['origin'],e['features'])
    # An NLI-scored unordered pair can carry Q in the reverse direction, too.
    for e in d['edges']:
        for i,j in [(e['i'],e['j']),(e['j'],e['i'])]:
            if (COND.search(d['units'][j]['text']) or SCOPE.search(d['units'][j]['text'])) and claim_ok(d['units'][i]) and qualifier_ok(d['units'][j]):
                raw=e if i==e['i'] else dict(e,nli_forward=e.get('nli_reverse'),nli_reverse=e.get('nli_forward'))
                add(i,j,'Q',raw,'scored_pair_retyped_Q')
    local=[]
    for j,b in enumerate(d['units']):
        if not qualifier_ok(b) or not(COND.search(b['text']) or SCOPE.search(b['text'])):continue
        for i,a in enumerate(d['units']):
            if i!=j and claim_ok(a) and 0<abs(a['sid']-b['sid'])<=2:
                score=2*bool(re.match(r'^(?:If|Unless|Under|When|Provided|Subject to)\b',b['text'],re.I))+float(d['sim'][i,j])+float(d['u'][i])
                local.append((score,i,j))
    for _,i,j in sorted(local,reverse=True)[:32]:
        if (i,j,'Q') not in out:add(i,j,'Q',None,'local_dependent_qualifier')
    # No model or NLI calls: only known observations and masked new proposals.
    return sorted(out.values(),key=lambda e:(e['kind'],e['i'],e['j']))
