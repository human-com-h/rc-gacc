"""Outcome-blind source grouping, exact-shingle overlap and four-way splits."""
import hashlib,re,unicodedata
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix

def normalized(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())
def digest_text(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()

class Union:
    def __init__(self,n):self.parent=list(range(n))
    def root(self,i):
        while self.parent[i]!=i:
            self.parent[i]=self.parent[self.parent[i]];i=self.parent[i]
        return i
    def join(self,i,j):self.parent[self.root(j)]=self.root(i)

def overlap_pairs(documents,threshold=.85,min_shared=20,block_size=128,progress=None):
    """Exhaustive sparse intersections of exact 5-word shingles, no MinHash.

    Globally assigned uint32 token IDs make five-token byte keys injective.
    Containment is intersection/min(shingle counts); threshold is preregistered.
    This is a textual collision criterion, not a proof of semantic independence.
    """
    vocabulary={};signatures=[];width=5;dtype=np.dtype('V20')
    for i,doc in enumerate(documents):
        tokens=re.findall(r'\w+',normalized(doc),re.UNICODE)
        ids=np.fromiter((vocabulary.setdefault(t,len(vocabulary)) for t in tokens),dtype=np.uint32,count=len(tokens))
        if len(ids)>=width:
            windows=np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(ids,width))
            sig=np.unique(windows.view(dtype).ravel())
        else:sig=np.array([],dtype=dtype)
        signatures.append(sig)
        if progress and i%500==0:progress(f'shingles {i}/{len(documents)}')
    sizes=np.array([len(s) for s in signatures],dtype=np.int64)
    keys=np.concatenate(signatures) if signatures else np.array([],dtype=dtype)
    unique,cols=np.unique(keys,return_inverse=True);del keys
    rows=np.repeat(np.arange(len(documents),dtype=np.int32),sizes)
    x=csr_matrix((np.ones(len(rows),dtype=np.int32),(rows,cols)),shape=(len(documents),len(unique)))
    del signatures,rows,cols,unique,vocabulary
    pairs=[];n=len(documents)
    for start in range(0,n,block_size):
        m=(x[start:start+block_size]@x.T).tocoo()
        ii=m.row+start;jj=m.col
        eligible=(ii<jj)&(m.data>=min_shared)&(m.data>=threshold*np.minimum(sizes[ii],sizes[jj]))
        for i,j,shared in zip(ii[eligible],jj[eligible],m.data[eligible],strict=True):
            pairs.append(dict(i=int(i),j=int(j),shared=int(shared),left_shingles=int(sizes[i]),right_shingles=int(sizes[j]),
                containment=float(shared/min(sizes[i],sizes[j])),jaccard=float(shared/(sizes[i]+sizes[j]-shared))))
        if progress and start%512==0:progress(f'overlap {start}/{n}')
    return pairs,dict(documents=n,vocabulary='global word IDs, exact five-token byte tuples',
        total_shingles=int(sizes.sum()),minimum_shared=min_shared,containment_threshold=threshold,exhaustive=True)

def freeze_groups(records,overlaps,seed=42):
    if seed!=42:raise ValueError('Registered pilot split requires seed42')
    u=Union(len(records));seen={}
    for i,r in enumerate(records):
        for key in r['group_keys']+[('normalized_source',r['normalized_sha256'])]:
            key=tuple(key)
            if key in seen:u.join(i,seen[key])
            seen[key]=i
    for p in overlaps:u.join(p['record_i'],p['record_j'])
    buckets=defaultdict(list)
    for i,r in enumerate(records):buckets[u.root(i)].append(r)
    groups=[]
    for rr in buckets.values():
        ids=sorted(r['record_id'] for r in rr)
        strata=sorted({r['stratum'] for r in rr})
        groups.append(dict(group_id='grp_'+digest_text('|'.join(ids)),record_ids=ids,
            stratum=strata[0] if len(strata)==1 else 'mixed',source_strata=strata,
            normalized_hashes=sorted({r['normalized_sha256'] for r in rr}),
            explored_members=sorted(r['record_id'] for r in rr if r['explored']),
            split='exploratory_quarantine' if any(r['explored'] for r in rr) else None))
    rng=np.random.default_rng(seed)
    for stratum in sorted({g['stratum'] for g in groups if g['split'] is None}):
        gg=sorted([g for g in groups if g['stratum']==stratum and g['split'] is None],key=lambda g:g['group_id'])
        order=rng.permutation(len(gg));n=len(gg);train_dev=int(.7*n);dev=int(.15*n);cal_end=int(.85*n)
        for rank,k in enumerate(order):
            gg[k]['split']='train' if rank<train_dev-dev else 'dev' if rank<train_dev else 'calibration' if rank<cal_end else 'final'
    return sorted(groups,key=lambda g:g['group_id'])
