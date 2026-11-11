"""Strict local reader aggregation: missing judgements never become negatives."""
from collections import defaultdict

def aggregate(expected,observed):
    key=lambda r:(r['family'],r['id'],r['method'],r['position'])
    plan={key(r):r for r in expected}
    if len(plan)!=len(expected):raise ValueError('Duplicate planned request')
    pred={}
    for row in observed:
        k=key(row)
        if k not in plan:raise ValueError('Unexpected prediction')
        if k in pred:raise ValueError('Duplicate prediction')
        pred[k]=row
    groups=defaultdict(list)
    for k,r in plan.items():groups[k[:3]].append((r,pred.get(k)))
    cases=[]
    for (family,sid,method),items in groups.items():
        result={'family':family,'id':sid,'method':method,'track':items[0][0]['track'],
                'expected_outputs':len(items),'observed_outputs':sum(p is not None for _,p in items)}
        valid=True;values=[];condition=None;answer=None;scope_mask=None;scope_required=False
        for ref,row in items:
            if family=='keep_semantic' and ref['judge']=='key':
                scope_required=ref.get('scope_mask_required',False) or 'scope_span_retained' in ref
                scope_mask=ref.get('scope_span_retained')
                if scope_mask is not None and type(scope_mask) is not bool:raise ValueError('Invalid scope retention mask')
            p=None if row is None else row.get('parsed')
            if not isinstance(p,dict):valid=False;continue
            if family=='keep_semantic':
                fields=['condition_judgement','overall_judgement'] if ref['judge']=='key' else ['judgement']
                if any(p.get(f) not in ('yes','no') for f in fields):valid=False;continue
                values.extend(p[f]=='yes' for f in fields)
                if ref['judge']=='key':condition=p['condition_judgement']=='yes'
            else:
                answer=p.get('answer')
                if answer not in ref['labels']:valid=False;continue
                values.append(answer==ref['gold'])
        result['complete_valid']=valid and result['expected_outputs']==result['observed_outputs']
        if family=='keep_semantic':
            result.update(all_rec=bool(all(values)) if result['complete_valid'] else None,
                          cond_rec=condition if result['complete_valid'] else None)
            if scope_required:
                result.update(raw_all_rec=result['all_rec'],raw_cond_rec=result['cond_rec'],
                              scope_span_retained=scope_mask,
                              condition_missing=(not scope_mask) if scope_mask is not None else None)
                if result['complete_valid']:
                    for metric in ['all_rec','cond_rec']:
                        raw=result[metric]
                        result[metric]=False if raw is False or scope_mask is False else (True if raw is True and scope_mask is True else None)
        else:
            result.update(correct=bool(all(values)) if result['complete_valid'] else None,answer=answer,
                          false_directional=(answer in ('AFFIRMED','NEGATED') if result['track'].endswith('_C') else answer=='UNCONDITIONAL') if result['complete_valid'] else None)
        cases.append(result)
    summary=[]
    for family,track,method in sorted({(x['family'],x['track'],x['method']) for x in cases}):
        rows=[x for x in cases if (x['family'],x['track'],x['method'])==(family,track,method)]
        valid=[x for x in rows if x['complete_valid']]
        s={'family':family,'track':track,'method':method,'expected_cases':len(rows),'complete_valid_cases':len(valid)}
        for metric in (['all_rec','cond_rec'] if family=='keep_semantic' else ['correct','false_directional']):
            v=[x[metric] for x in valid if x[metric] is not None]
            s[metric]=sum(v)/len(v) if v else None;s[metric+'_n']=len(v)
        summary.append(s)
    return cases,summary
