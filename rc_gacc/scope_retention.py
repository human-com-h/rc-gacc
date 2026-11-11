"""Source-span evidence veto, independent of a judge's condition labels."""


def scope_span_mask(source_length, retained_spans, required_groups):
    """Each required scope part has one or more acceptable source-span alternatives.

    Missing annotations return None, never a fabricated condition-missing label.
    Coordinates refer to the original source, not text injected into judge prompts.
    """
    if not required_groups:
        return None
    def valid(span):
        return len(span) == 2 and all(type(v) is int for v in span) and 0 <= span[0] < span[1] <= source_length
    if any(not valid(s) for s in retained_spans) or any(not g or any(not valid(s) for s in g) for g in required_groups):
        raise ValueError('Invalid source-span annotation')
    merged = []
    for lo, hi in sorted(retained_spans):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    return all(any(any(a <= lo and hi <= b for a, b in merged) for lo, hi in alternatives)
               for alternatives in required_groups)


def joint_condition_result(mask, parsed):
    labels = parsed if isinstance(parsed, dict) else {}
    judge = (labels['condition_judgement'] == 'yes' and labels['overall_judgement'] == 'yes'
             if all(labels.get(k) in ('yes', 'no') for k in ('condition_judgement', 'overall_judgement')) else None)
    joint = False if mask is False or judge is False else (True if mask is True and judge is True else None)
    return dict(scope_span_retained=mask, condition_missing=(not mask) if mask is not None else None,
                judge_condition_and_claim=judge, joint_conditional_retention=joint)
