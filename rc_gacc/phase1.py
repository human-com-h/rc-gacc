import json, os, re
from pathlib import Path

SCHEMA_VERSION = 'rc-gacc-phase1-v1'

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temp, path)

def inline_render(units, selected, bindings, tokenizer, budget):
    """Triggered markers, no source-ID tax without bindings; exact hard accounting.

    bindings contains only activated, selected pairs. A node is emitted once;
    only disjoint pairs are compacted in v1. Overlapping pairs stay unbound and
    are explicitly reported, rather than duplicating free evidence.
    """
    chosen = set(selected)
    selected_pairs = [b for b in bindings if b['i'] in chosen and b['j'] in chosen]
    plain = '\n'.join(units[i]['text'] for i in sorted(chosen))
    emitted, arranged, accepted, skipped = set(), [], [], []
    starts = {}
    for b in sorted(selected_pairs, key=lambda b:(min(b['i'], b['j']), b['kind'])):
        if any(i in emitted for i in (b['i'], b['j'])):
            skipped.append(dict(**b, reason='overlapping_pair_requires_general_renderer'))
            continue
        emitted.update([b['i'], b['j']])
        # A qualifier saying "the preceding claim" must stay AFTER its claim.
        # Ordinary leading if/when clauses are kept before the claim.
        anaphoric = bool(re.match(r'^The preceding claim\b', units[b['j']]['text']))
        order = [b['j'], b['i']] if b['kind']=='Q' and not anaphoric else [b['i'], b['j']]
        starts[min(order)] = (b, order)
        accepted.append(b)
    emitted = set()
    for i in sorted(chosen):
        if i in starts:
            b, order = starts[i]
            arranged.append(b['kind']+': '+units[order[0]]['text'])
            arranged.append(units[order[1]]['text'])
            emitted.update(order)
        elif i not in emitted and not any(i in (b['i'], b['j']) for b in accepted):
            arranged.append(units[i]['text'])
    text = '\n'.join(arranged) if accepted else plain
    count = lambda t:len(tokenizer.encode(t, disallowed_special=()))
    if count(text) > budget:
        raise ValueError('Inline serialization exceeds hard budget; no silent truncation')
    return dict(schema_version=SCHEMA_VERSION, reader_text=text, tokens=count(text), budget=budget,
        plain_tokens=count(plain), total_binding_delta_tokens=count(text)-count(plain),
        labels=len(accepted), active_bindings=accepted, unbound_overlaps=skipped,
        source_ids=[units[i]['sid'] for i in sorted(chosen)],
        provenance_contract='Only reader_text is sent as compressed evidence; metadata must not be silently appended.')
