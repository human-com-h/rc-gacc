"""Explicit retained-edge register, with node and edge token costs."""
def line(unit):
    return f"[s{unit['sid']}] {unit['text']}\n"

def relation_line(units,edge):
    a,b=units[edge.i]['sid'],units[edge.j]['sid']
    if edge.kind=='C':return f'[CONFLICT_PAIR s{a} <-> s{b}]\n'
    if edge.kind=='Q':return f'[SCOPE_LIMIT claim=s{a} qualifier=s{b}]\n'
    raise ValueError('Phase 0 interface supports C/Q; other types require their own semantics')

def cost_model(units,edges,count):
    pairs=[e for e in edges if e.kind in ('C','Q')]
    # The same worst-view cost is used for both renders in iso-content ablations.
    nodes=[count(line(u))+count('s'+str(u['sid'])+',')+1 for u in units]
    relation_costs=[count(relation_line(units,e))+1 for e in pairs]
    overhead=count('CORE\nCERTIFICATE\nRELATIONS\nSKELETON \n')
    return nodes,relation_costs,overhead

def render(units,selected,edges,style):
    chosen=set(selected)
    if style=='flat':return ''.join(line(units[i]) for i in sorted(chosen))
    active=[e for e in edges if e.kind in ('C','Q') and e.i in chosen and e.j in chosen]
    cert={i for e in active for i in (e.i,e.j)}
    core=chosen-cert
    text=''
    if core:text+='CORE\n'+''.join(line(units[i]) for i in sorted(core))
    if cert:
        text+='CERTIFICATE\nRELATIONS\n'+''.join(relation_line(units,e) for e in active)
        text+=''.join(line(units[i]) for i in sorted(cert))
    if chosen:text+='SKELETON '+','.join('s'+str(units[i]['sid']) for i in sorted(chosen))+'\n'
    return text
