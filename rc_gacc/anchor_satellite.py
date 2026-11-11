"""Public-source-only candidate and pair pre-screen. No oracle inputs."""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

MARK=re.compile(r'\b(?:if|unless|only|except|provided|under|during|before|after|however|although|whereas|but|not|never|despite|without|until|when|since)\b',re.I)
STOP=set('the a an and or of to for in on at is are was were it this that with from by as be have has had can will would should which their its they these those more most only not'.split())
def terms(s):return set(re.findall(r'[a-z][a-z0-9_]{2,}',s.lower()))-STOP
def ranks(units,goal):
    sub=[s.strip() for s in re.split(r'[;.!?]|\band\b',goal) if s.strip()][:8] or [goal]
    try:
        mat=TfidfVectorizer(token_pattern=r'(?u)\b\w+\b',ngram_range=(1,2)).fit_transform([u['text'] for u in units]+[goal]+sub)
        z=(mat[:len(units)]@mat[len(units):].T).toarray()
        score=.6*z[:,0]+.4*z[:,1:].max(axis=1)
    except ValueError:score=np.zeros(len(units))
    return sorted(range(len(units)),key=lambda i:(-score[i],i)),score

def preselect(units,goal,cap=120):
    if cap<4: return units[:cap]
    order,score=ranks(units,goal);main_cap=min(cap,int(round(cap*88/120)))
    core=order[:main_cap];chosen=set(core);anchors=order[:32];tt=[terms(u['text']) for u in units]
    proposals=[]
    for i in anchors:
        for j in range(max(0,i-2),min(len(units),i+3)):
            if i==j or not MARK.search(units[j]['text']):continue
            common=len(tt[i]&tt[j]);priority=3+common/max(1,len(tt[i]|tt[j]))+float(score[i])
            proposals.append((priority,i,j,'local_scope'))
        for j in range(len(units)):
            if i==j or not MARK.search(units[j]['text']):continue
            common=tt[i]&tt[j]
            if len(common)<2:continue
            priority=1+len(common)/max(1,len(tt[i]|tt[j]))+float(score[i])
            proposals.append((priority,i,j,'entity_competitor'))
    for _,i,j,kind in sorted(proposals,key=lambda x:(-x[0],x[1],x[2])):
        if len(chosen)>=cap:break
        chosen.add(j)
    for i in order:
        if len(chosen)>=cap:break
        chosen.add(i)
    return [units[i] for i in sorted(chosen)]

def pair_order(units,similarity,cap=24):
    n=len(units);tt=[terms(u['text']) for u in units];proposals=[]
    for i in range(n):
        for j in range(i+1,n):
            gap=abs(units[i]['sid']-units[j]['sid']);common=len(tt[i]&tt[j]);cue=bool(MARK.search(units[i]['text']) or MARK.search(units[j]['text']))
            if cue and ((gap<=2) or common>=2):
                proposals.append((3*(gap<=2)+common/max(1,len(tt[i]|tt[j]))+float(similarity[i,j]),i,j))
    chosen=[]
    for _,i,j in sorted(proposals,key=lambda z:(-z[0],z[1],z[2]))[:max(1,cap*2//3)]:chosen.append((i,j))
    ranked=sorted(((i,j) for i in range(n) for j in range(i+1,n)),key=lambda p:(-float(similarity[p[0],p[1]]),p))
    for p in ranked:
        if len(chosen)>=cap:break
        if p not in chosen:chosen.append(p)
    return chosen
