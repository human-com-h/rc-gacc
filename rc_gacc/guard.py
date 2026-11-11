"""The Guarded working point."""

def guarded(safe, swap):
    """Switch only when a credited gate edge is newly paired relative to Safe."""
    kept = set(safe['selected'])
    return any(not {edge['i'],edge['j']}.issubset(kept) for edge in swap['protection'])
