"""Squad selection and the transfer decision.

Two modes:
  build()     — best 15 from scratch (season start, or a wildcard)
  transfers() — given your current 15, price up 0/1/2/3 moves against the -4 hit

Both jointly pick the starting XI and captain for every gameweek in the horizon,
so the 15 is built around lineups that are actually legal rather than 15 names
that happen to score well.
"""
import pandas as pd, pulp

BUDGET = 100.0
DECAY = 0.87          # a gameweek further out counts less — you can still transfer
BENCH_W = 0.12        # what an outfield bench slot is worth
BENCH_GK_W = 0.03
HIT = 4.0
SQUAD = [('GK', 2), ('DEF', 5), ('MID', 5), ('FWD', 3)]


def vice(xi, cap, g, xp, pstart):
    """Pick the vice-captain and price the insurance he provides.

    The armband only moves if the captain plays *zero* minutes, so the vice is
    worth P(captain doesn't play) x his own expected points — usually a fraction
    of a point, occasionally decisive. Given the captain is fixed, that
    probability is a constant, so the best vice is simply the highest-scoring
    other starter.
    """
    others = [i for i in xi if i != cap]
    if not others:
        return dict(vice=None, insurance=0.0, xp=round(
            sum(xp.get((i, g), 0) for i in xi) + xp.get((cap, g), 0), 2))
    vc = max(others, key=lambda i: xp.get((i, g), 0))
    p_cap_plays = pstart.get((cap, g), 1.0)
    ins = (1 - p_cap_plays) * xp.get((vc, g), 0)
    return dict(vice=int(vc), insurance=round(ins, 3),
                xp=round(sum(xp.get((i, g), 0) for i in xi)
                         + xp.get((cap, g), 0) + ins, 2))


def _solve(proj, tot, gws, budget=BUDGET, force=(), ban=(), keep=None,
           max_out=None, min_start=None, eo_penalty=0.0, time_limit=300):
    pool = tot[((tot.xp_h > 3.0) | (tot.price <= 4.5)) & (tot.status != 'u')]
    if keep:
        pool = pd.concat([pool, tot[tot.id.isin(keep)]]).drop_duplicates('id')
    if min_start is not None:
        pool = pool[(pool.p_start >= min_start) | (pool.id.isin(keep or []))]
    pl = pool.set_index('id')
    ids = list(pl.index)
    xp = {(r.id, r.gw): r.xp for r in proj[proj.id.isin(ids)].itertuples()}
    pstart = {(r.id, r.gw): r.p_start for r in proj[proj.id.isin(ids)].itertuples()}
    pos, price, club = pl.pos.to_dict(), pl.price.to_dict(), pl.team_name.to_dict()
    owned = pl.selected_by.to_dict()

    m = pulp.LpProblem('fpl', pulp.LpMaximize)
    sq = pulp.LpVariable.dicts('sq', ids, cat='Binary')
    st = pulp.LpVariable.dicts('st', [(i, g) for i in ids for g in gws], cat='Binary')
    cp = pulp.LpVariable.dicts('cp', [(i, g) for i in ids for g in gws], cat='Binary')

    obj = []
    for k, g in enumerate(gws):
        w = DECAY ** k
        for i in ids:
            e = xp.get((i, g), 0.0)
            bw = BENCH_GK_W if pos[i] == 'GK' else BENCH_W
            obj += [w * e * st[(i, g)], w * e * cp[(i, g)],
                    w * e * bw * (sq[i] - st[(i, g)])]
    if eo_penalty:
        # chasing rank rather than protecting it: shade away from the template
        obj += [-eo_penalty * (owned[i] / 100.0) * sq[i] for i in ids]
    m += pulp.lpSum(obj)

    m += pulp.lpSum(sq[i] for i in ids) == 15
    for p, n in SQUAD:
        m += pulp.lpSum(sq[i] for i in ids if pos[i] == p) == n
    m += pulp.lpSum(price[i] * sq[i] for i in ids) <= budget
    for c in set(club.values()):
        m += pulp.lpSum(sq[i] for i in ids if club[i] == c) <= 3
    for i in force:
        if i in sq:
            m += sq[i] == 1
    for i in ban:
        if i in sq:
            m += sq[i] == 0
    if keep is not None and max_out is not None:
        m += 15 - pulp.lpSum(sq[i] for i in ids if i in keep) <= max_out
    for g in gws:
        m += pulp.lpSum(st[(i, g)] for i in ids) == 11
        m += pulp.lpSum(st[(i, g)] for i in ids if pos[i] == 'GK') == 1
        m += pulp.lpSum(st[(i, g)] for i in ids if pos[i] == 'DEF') >= 3
        m += pulp.lpSum(st[(i, g)] for i in ids if pos[i] == 'MID') >= 2
        m += pulp.lpSum(st[(i, g)] for i in ids if pos[i] == 'FWD') >= 1
        m += pulp.lpSum(cp[(i, g)] for i in ids) == 1
        for i in ids:
            m += st[(i, g)] <= sq[i]
            m += cp[(i, g)] <= st[(i, g)]

    m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    if pulp.LpStatus[m.status] != 'Optimal':
        return None

    chosen = [i for i in ids if sq[i].value() > 0.5]
    res = pl.loc[chosen].reset_index()
    res['xp_next'] = [xp.get((i, gws[0]), 0) for i in res.id]
    res['xp_h'] = [sum(xp.get((i, g), 0) for g in gws) for i in res.id]
    res['start'] = [st[(i, gws[0])].value() > 0.5 for i in res.id]
    res['captain'] = [cp[(i, gws[0])].value() > 0.5 for i in res.id]
    lineups = {}
    for g in gws:
        xi = [int(i) for i in ids if st[(i, g)].value() > 0.5]
        cap = int(next(i for i in ids if cp[(i, g)].value() > 0.5))
        lineups[int(g)] = dict(xi=xi, captain=cap,
                               **vice(xi, cap, g, xp, pstart))
    raw = sum(l['xp'] for l in lineups.values())          # plain projected points
    weighted = sum(l['xp'] * (DECAY ** k) for k, l in enumerate(lineups.values()))
    return dict(squad=res, lineups=lineups, raw=round(raw, 2),
                weighted=round(weighted, 2),
                out=sorted(set(keep) - set(chosen)) if keep else [],
                cost=round(float(res.price.sum()), 1))


def build(proj, tot, gws, **kw):
    return _solve(proj, tot, gws, **kw)


def transfers(proj, tot, gws, held, bank, free, max_hits=2):
    """Rank 0..(free+max_hits) transfers by expected points net of the -4 hits."""
    held = set(held)
    budget = round(float(tot[tot.id.isin(held)].price.sum()) + bank, 1)
    runs = []
    for n in range(0, free + max_hits + 1):
        r = _solve(proj, tot, gws, budget=budget, keep=held, max_out=n)
        if not r:
            continue
        made = len(r['out'])
        r['transfers'] = made
        r['hit'] = max(0, made - free) * HIT
        r['net'] = round(r['raw'] - r['hit'], 2)
        runs.append(r)
    # deduplicate: a 2-transfer solve may return the same squad as the 1-transfer one
    seen, uniq = set(), []
    for r in sorted(runs, key=lambda r: r['transfers']):
        key = tuple(sorted(r['squad'].id))
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    best = max(uniq, key=lambda r: r['net'])
    return dict(options=uniq, best=best,
                hold=next(r for r in uniq if r['transfers'] == 0))
