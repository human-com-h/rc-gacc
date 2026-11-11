"""Shared-budget baselines and relation-constrained scipy MILP.

No encoder, reader, or network calls occur in this module.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class Edge:
    i: int
    j: int
    kind: str  # C: symmetric unresolved; Q: claim -> qualifier; T: newer -> older
    confidence: float = 1.0


def pair_loss(p, utility, edges, tau_u=0.4, tau_p=0.6):
    """Prototype loss definition; utility must be a frozen teacher signal.

    This evaluable specification is not a claim that a scorer was trained.
    The learnable head and independent calibration belong to Phase 2.
    """
    return sum(abs(p[e.i] - p[e.j]) + max(0.0, tau_p - min(p[e.i], p[e.j]))
               for e in edges if e.kind == "C" and max(utility[e.i], utility[e.j]) > tau_u)


def objective(selected, utility, coverage, edges, relation_bonus=0.75):
    s = set(selected)
    value = sum(utility[i] for i in s)
    if s and coverage.size:
        value += 0.5 * np.max(coverage[:, sorted(s)], axis=1).sum()
    value += relation_bonus * sum(max(utility[e.i], utility[e.j]) * e.confidence
                                  for e in edges if e.kind in ("C", "Q") and e.i in s and e.j in s)
    return float(value)


def solve_milp(costs, budget, utility, coverage, edges=(), enforce=True,
               relax_conflicts=False, relation_bonus=0.75, time_limit=5.0, edge_costs=None):
    """Maximize utility + max-coverage + complete-relation rewards.

    Coverage is piecewise linearized exactly with binary assignment variables;
    each target chooses at most one selected witness. Cost is the caller's
    conservative serialization cost; exact rendered tokens are checked outside.
    Q is always hard. Optional C slack penalizes incomplete pairs, never budget.
    """
    started = perf_counter()
    costs, u, a = np.asarray(costs, float), np.asarray(utility, float), np.asarray(coverage, float)
    n = len(u)
    if a.size == 0:
        a = np.zeros((0, n))
    k = len(a)
    pairs = [e for e in edges if e.kind in ("C", "Q")]
    edge_costs = np.zeros(len(pairs)) if edge_costs is None else np.asarray(edge_costs,float)
    if len(edge_costs)!=len(pairs) or np.any(edge_costs<0):raise ValueError('Invalid edge serialization costs')
    conflicts = [e for e in edges if e.kind == "C"] if enforce and relax_conflicts else []
    # x[n], assignment[k*n], z[len(pairs)], xi[len(conflicts)]
    z0 = n + k * n
    slack0 = z0 + len(pairs)
    size = slack0 + len(conflicts)
    c = np.zeros(size)
    c[:n] = -u
    c[n:z0] = -0.5 * a.ravel()
    for t, e in enumerate(pairs):
        c[z0+t] = -relation_bonus * max(u[e.i], u[e.j]) * e.confidence
    c[slack0:] = 2.0  # explicit, fixed incomplete-pair penalty
    rows, lower, upper = [], [], []

    def add(vals, lo=-np.inf, hi=np.inf):
        row = np.zeros(size)
        for col, value in vals.items():
            row[col] += value
        rows.append(row)
        lower.append(lo)
        upper.append(hi)

    budget_row=dict(enumerate(costs))
    budget_row.update({z0+t:float(value) for t,value in enumerate(edge_costs)})
    add(budget_row, hi=max(0, budget))
    for target in range(k):
        add({n+target*n+i: 1 for i in range(n)}, hi=1)
        for i in range(n):
            add({n+target*n+i: 1, i: -1}, hi=0)
    for t, e in enumerate(pairs):
        z = z0 + t
        add({z: 1, e.i: -1}, hi=0)
        add({z: 1, e.j: -1}, hi=0)
        add({e.i: 1, e.j: 1, z: -1}, hi=1)
    ci = 0
    if enforce:
        for e in edges:
            if e.kind == "C":
                if relax_conflicts:
                    v = slack0 + ci
                    ci += 1
                    add({e.i: 1, e.j: -1, v: -1}, hi=0)
                    add({e.j: 1, e.i: -1, v: -1}, hi=0)
                else:
                    add({e.i: 1, e.j: -1}, lo=0, hi=0)
            elif e.kind == "Q":
                add({e.i: 1, e.j: -1}, hi=0)
    integrality = np.zeros(size)
    integrality[:n] = 1
    # For binary x, assignment LPs attain max coverage at a selected witness,
    # AND constraints fix z exactly, and the C slack optimum is already 0 or 1.
    result = milp(c, integrality=integrality, bounds=Bounds(0, 1),
                  constraints=LinearConstraint(np.asarray(rows), lower, upper),
                  options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if result.x is None:
        selected = []  # certified feasible empty fallback, never called optimal
    else:
        selected = np.flatnonzero(result.x[:n] > 0.5).tolist()
    violations = [{"i":e.i,"j":e.j,"kind":e.kind} for e in edges
                  if (e.kind == "C" and ((e.i in selected) != (e.j in selected)))
                  or (e.kind == "Q" and e.i in selected and e.j not in selected)]
    selected_set=set(selected)
    total_cost=sum(costs[selected])+sum(edge_costs[t] for t,e in enumerate(pairs) if e.i in selected_set and e.j in selected_set)
    if total_cost > max(0, budget) + 1e-7:
        raise RuntimeError("MILP returned an over-budget solution")
    if enforce and (any(e["kind"] == "Q" for e in violations) or (violations and not relax_conflicts)):
        raise RuntimeError("MILP returned a relation-infeasible solution")
    return selected, {"status":int(result.status),"message":result.message,
                      "proven_optimal":bool(result.status == 0 and result.x is not None),
                      "mip_gap":getattr(result,"mip_gap",None),
                      "mip_dual_bound":getattr(result,"mip_dual_bound",None),
                      "minimized_objective":getattr(result,"fun",None),
                      "seconds":perf_counter()-started,"violations":violations,
                      "relaxed":relax_conflicts,"cost_model":"additive_nodes_and_edge_register",
                      "selected_proxy_cost":float(total_cost)}


def rank_select(costs, budget, utility, similarity, method, edges=()):
    u = np.asarray(utility).copy()
    if method == "surface_both":
        for e in edges:
            if e.kind in ("C", "Q"):
                bonus = 0.75 * max(utility[e.i], utility[e.j]) * e.confidence
                u[e.i] += bonus
                u[e.j] += bonus
    selected, remaining, spent = [], set(range(len(u))), 0
    while remaining:
        feasible = [i for i in remaining if spent + costs[i] <= budget]
        if not feasible:
            break
        def score(i):
            if method == "mmr" and selected:
                return 0.7 * u[i] - 0.3 * max(similarity[i,j] for j in selected)
            return u[i]
        best = max(feasible, key=lambda i:(score(i), -i))
        selected.append(best)
        remaining.remove(best)
        spent += costs[best]
    return sorted(selected), {"status":"ranking","proven_optimal":False,"violations":[]}
