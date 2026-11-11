import re
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from .selector import Edge
TOK = tiktoken.get_encoding("cl100k_base")

def count(text):
    return len(TOK.encode(text,disallowed_special=()))

def words(text):
    return set(re.findall(r"[a-z0-9]+",text.lower()))

def chunks(text, max_tokens=96):
    """Extract source units; keep exact source substrings, no generated claims."""
    pieces = re.split(r"(?<=[.!?。！？])\s+|\n+",text)
    result=[]
    for piece in pieces:
        piece=piece.strip()
        if not piece:
            continue
        # Split a leading explicit time/scope clause without consulting gold keys.
        m=re.match(r"^((?:In (?:the )?(?:\d{4}s?|early \d{4}s?|late \d{4}s?)|Under [^,]{3,80}|If [^,]{3,80}|When [^,]{3,80})[,;])\s+(.+)",piece)
        subs=[m.group(1),m.group(2)] if m else [piece]
        for sub in subs:
            if count(sub)<=max_tokens:
                result.append(sub)
            else:
                # Word boundaries preserve the original text rather than decoding partial UTF-8 tokens.
                spans=list(re.finditer(r"\S+\s*",sub))
                start=0
                for j in range(1,len(spans)+1):
                    end=spans[j-1].end()
                    if count(sub[spans[start].start():end])>max_tokens and j-1>start:
                        result.append(sub[spans[start].start():spans[j-2].end()].strip())
                        start=j-1
                if start<len(spans):
                    result.append(sub[spans[start].start():].strip())
    return result

def make_units(text):
    return [{"sid":i,"text":s} for i,s in enumerate(chunks(text))]

def lexical_prune(units,goal,cap):
    texts=[u['text'] for u in units]
    if len(units)<=cap:
        return units
    try:
        matrix=TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",ngram_range=(1,2)).fit_transform(texts+[goal])
        scores=(matrix[:-1]@matrix[-1].T).toarray().ravel()
    except ValueError:
        scores=np.zeros(len(units))
    # Reserve immediate predecessors to keep candidate-level scope closure available.
    chosen=set()
    for i in sorted(range(len(units)),key=lambda j:(-scores[j],j)):
        group={i}
        if i and re.match(r"^(?:In (?:the )?\d{4}|If |When |Under )",units[i-1]['text']):
            group.add(i-1)
        if i+1<len(units) and re.match(r"^(?:This|That|The preceding) .*(?:only (?:if|when|under)|conditional on)",units[i+1]['text'],re.I):
            group.add(i+1)
        if len(chosen|group)<=cap:
            chosen|=group
        if len(chosen)>=cap:
            break
    return [units[i] for i in sorted(chosen)]

def extract_edges(units):
    """Auditable high-precision rules only; no gold evidence, question answer, NLI model."""
    edges=[]
    for i,u in enumerate(units):
        s=u['text']
        if i and units[i-1]['sid']==u['sid']-1:
            prev=units[i-1]['text']
            if re.match(r"^(?:In (?:the )?\d{4}|If |When |Under )",prev) and prev.endswith((',', ';')):
                edges.append(Edge(i,i-1,"Q",0.9))
            if re.match(r"^(?:This|That|The preceding) .*(?:only (?:if|when|under)|conditional on)",s,re.I):
                edges.append(Edge(i-1,i,"Q",0.95))
        for j in range(i):
            t=units[j]['text']
            a,b=words(s),words(t)
            overlap=len(a&b)/max(1,len(a|b))
            opposite=(('approved' in a and 'rejected' in b) or ('rejected' in a and 'approved' in b))
            neg=lambda z:bool(re.search(r"\b(?:not|never|cannot)\b",z.lower()))
            if overlap>0.65 and (opposite or (neg(s)!=neg(t) and overlap>0.8)):
                edges.append(Edge(j,i,"C",0.95 if opposite else 0.75))
    # Avoid duplicate C edges or multiple rule firings.
    return list({(e.i,e.j,e.kind):e for e in edges}.values())
