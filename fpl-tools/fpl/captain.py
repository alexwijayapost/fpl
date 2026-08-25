"""Captaincy as a variance decision, not just a maximum.

Expected points alone answers the wrong question in a head-to-head league. There
you play one opponent per week: you need a *ceiling* often enough to beat them,
and the cost of a blank is capped at losing one match. In a classic league the
same blank costs you rank against eleven million managers at once.

So this returns a distribution, not a number: the chance of a haul, the chance of
a blank, and an effective-ownership-adjusted view of what each choice does to
your rank if it lands or misses.
"""
import numpy as np
from scipy.stats import poisson

GOAL_PTS = {'GK': 10, 'DEF': 6, 'MID': 5, 'FWD': 4}
CS_PTS = {'GK': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}
HAUL, BLANK = 10, 2


def distribution(row, max_goals=5, max_assists=4):
    """Point mass function for one player in one fixture."""
    pos = row['pos']
    ps = float(row['p_start'])
    lg, la = float(row['xg']), float(row['xa'])
    pcs = float(row['p_cs'])
    bonus = float(row.get('bonus90', 0.0))
    dc = float(row.get('pts_dc', 0.0)) / max(ps, 1e-6)   # back out per-start value

    pmf = {}

    def add(pts, p):
        pmf[pts] = pmf.get(pts, 0.0) + p

    # did not start: appearance 0 or a substitute cameo
    add(0, (1 - ps) * 0.65)
    add(1, (1 - ps) * 0.35)

    for g in range(max_goals + 1):
        pg = poisson.pmf(g, lg) if g < max_goals else 1 - poisson.cdf(max_goals - 1, lg)
        for a in range(max_assists + 1):
            pa = (poisson.pmf(a, la) if a < max_assists
                  else 1 - poisson.cdf(max_assists - 1, la))
            for cs, pc in ((1, pcs), (0, 1 - pcs)):
                base = 2 + g * GOAL_PTS[pos] + a * 3 + cs * CS_PTS[pos]
                # bonus and DefCon are small and near-deterministic at this
                # resolution; fold them in as a rounded expectation
                pts = int(round(base + bonus + dc))
                add(pts, ps * pg * pa * pc)

    total = sum(pmf.values())
    return {k: v / total for k, v in sorted(pmf.items())}


def summarise(row):
    pmf = distribution(row)
    ev = sum(k * v for k, v in pmf.items())
    return dict(
        xp=round(ev, 2),
        p_haul=round(sum(v for k, v in pmf.items() if k >= HAUL), 3),
        p_blank=round(sum(v for k, v in pmf.items() if k <= BLANK), 3),
        p_double_digit=round(sum(v for k, v in pmf.items() if k >= HAUL), 3),
        ceiling=max(k for k, v in pmf.items() if v > 0.02))


def vice_rank(proj_next, xi_ids, captain_id, top=5):
    """Vice-captain options, ranked by the insurance they actually buy.

    The vice only ever scores if the captain plays no minutes at all. So his
    value is P(captain absent) x his own expected points — and P(captain absent)
    is the same number whoever you pick, which makes this a straight ranking by
    the vice's own expected points among the other ten starters.

    Kickoff order does not matter: FPL applies the switch automatically once the
    gameweek finishes, so a vice playing on Friday still covers a captain who
    sits out on Sunday. What does matter is sharing a fixture with the captain —
    a postponement would take out both at once — so same-match candidates are
    flagged and broken against when two options are close.
    """
    xi = proj_next[proj_next.id.isin(xi_ids)]
    cap_row = xi[xi.id == captain_id]
    if cap_row.empty:
        return None
    p_cap = float(cap_row.p_start.iloc[0])
    p_absent = 1 - p_cap
    cap_team, cap_opp = cap_row.team_name.iloc[0], cap_row.opp.iloc[0]

    opts = []
    for _, r in xi[xi.id != captain_id].nlargest(top, 'xp').iterrows():
        same = r['team_name'] in (cap_team, cap_opp)
        opts.append(dict(
            id=int(r['id']), name=r['name'], team=r['team_name'], pos=r['pos'],
            opp=r['opp'], home=bool(r['home']), xp=round(float(r['xp']), 2),
            p_start=round(float(r['p_start']), 3), same_match=bool(same),
            insurance=round(p_absent * float(r['xp']), 2)))
    if not opts:
        return None
    # rank by insurance, but only let the shared-fixture tiebreak apply when the
    # options are genuinely close — it is a small risk, not a large one
    top_ins = max(o['insurance'] for o in opts)
    opts.sort(key=lambda o: (-o['insurance'],
                             o['same_match'] and o['insurance'] > top_ins - 0.05))
    best = min(opts, key=lambda o: (o['same_match'] if
                                    o['insurance'] > top_ins - 0.05 else False,
                                    -o['insurance']))
    warn = ('' if not best['same_match'] else
            ' He is in the same match as your captain, so a postponement would '
            'cost you both — worth a manual swap if that ever looks likely.')
    return dict(
        options=opts, pick=best['name'], p_captain_absent=round(p_absent, 3),
        insurance=best['insurance'],
        note=(f"Vice-captain: {best['name']}. Your captain is "
              f"{round(p_cap*100)}% to play, so the armband moves about "
              f"{round(p_absent*100)}% of the time — worth roughly "
              f"{best['insurance']:.1f} points a week. It is real but small, so "
              f"take the most certain starter among your best players rather "
              f"than chasing a ceiling here." + warn))


def rank(proj_next, top=10, eo_scale=1.6):
    """Captain options for the next gameweek, both ways of reading them.

    eo_scale converts ownership into effective ownership: a heavily owned player
    is captained more often than he is owned, so his effective ownership runs
    roughly 1.5-2x his raw ownership among the managers who own him.
    """
    cands = proj_next.nlargest(top * 3, 'xp')
    rows = []
    for _, r in cands.iterrows():
        s = summarise(r)
        eo = min(float(r['selected_by']) * eo_scale, 100.0)
        rows.append(dict(
            id=int(r['id']), name=r['name'], team=r['team_name'], pos=r['pos'],
            opp=r['opp'], home=bool(r['home']), owned=float(r['selected_by']),
            eo=round(eo, 1), **s,
            # what captaining him does to your rank relative to the field
            swing_up=round(s['xp'] * 2 * (1 - eo / 100), 2),
            swing_down=round(-s['xp'] * 2 * (eo / 100), 2)))
    rows.sort(key=lambda r: -r['xp'])
    safe = max(rows[:top], key=lambda r: r['xp'] - 0.5 * r['p_blank'] * 10)
    aggressive = max(rows[:top], key=lambda r: r['p_haul'] * (1 - r['eo'] / 100))
    return dict(options=rows[:top], safest=safe['name'], highest_ceiling=aggressive['name'],
                note=('Same pick either way.' if safe['name'] == aggressive['name'] else
                      f"Classic league says {safe['name']}; head-to-head says "
                      f"{aggressive['name']} — more likely to haul, and fewer of your "
                      f"rivals own him."))
