#!/usr/bin/env python3
"""FPL weekly run.

    python3 run.py                    # normal weekly run
    python3 run.py --fresh            # ignore my squad, build the best 15 from scratch
    python3 run.py --local bootstrap-static.json

Writes docs/index.html (the dashboard), plus CSVs under out/ and a dated
price/ownership snapshot under data/snapshots/ that accumulates over the season.
"""
import argparse, json, os, sys, datetime as dt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpl import sources, model, optimise, captain

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT, DOCS = (os.path.join(HERE, d) for d in ('data', 'out', 'docs'))
CFG = json.load(open(os.path.join(HERE, 'config.json')))
HORIZON = CFG.get('horizon', 6)


def rd(name):
    p = os.path.join(DATA, name)
    return pd.read_csv(p, low_memory=False) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fresh', action='store_true')
    ap.add_argument('--local', default=os.path.join(HERE, 'bootstrap-static.json'))
    ap.add_argument('--gw', type=int, help='override the gameweek (testing)')
    args = ap.parse_args()
    for d in (OUT, DOCS):
        os.makedirs(d, exist_ok=True)

    print('fetching...')
    players, fixtures, teams, prov = sources.load(DATA, args.local)
    prev_gw = rd('prev_merged_gw.csv')
    prev_players = rd('prev_players_raw.csv')
    prev_teams = rd('prev_teams.csv')
    cur_gw = rd('merged_gw.csv')

    ts = model.team_strength(teams, prev_gw, prev_teams, cur_gw)
    cur = model.prepare(players, teams, prev_players, prev_teams, ts)
    cur = model.start_probability(cur, cur_gw)

    # Which gameweek are we planning for? Trust FPL's own events list first.
    # Inferring it from fixture 'finished' flags is what broke this before: when
    # the fixtures fetch falls back to the stale preseason mirror, every match
    # looks unplayed, the run thinks it is still pre-GW1, and it rebuilds the
    # squad from scratch every day instead of advising on transfers.
    gw = args.gw or prov.get('gw_next')
    if args.gw:
        print(f'  gameweek {gw} (forced via --gw)')
    elif gw:
        print(f'  gameweek {gw} from FPL events (current: {prov.get("gw_current")})')
    else:
        unplayed = fixtures[~fixtures.finished.fillna(False)]
        gw = int(unplayed.event.dropna().min())
        print(f'  WARNING: no events data, inferred gameweek {gw} from fixtures')
    gw = int(gw)
    gws = [g for g in range(gw, gw + HORIZON) if g <= 38]
    print(f'  horizon {gws[0]}-{gws[-1]}')

    proj, tot = model.project(cur, ts, fixtures, gws)
    proj.to_csv(f'{OUT}/projections_gw.csv', index=False)
    tot.to_csv(f'{OUT}/projections_total.csv', index=False)

    # ---- who do we currently own?
    held = None
    if not args.fresh:
        mine = sources.my_team(CFG.get('entry_id'), gw - 1,
                               CFG.get('free_transfers', 1))
        sq_file = os.path.join(HERE, 'my_squad.json')
        if not mine and os.path.exists(sq_file):
            j = json.load(open(sq_file))
            mine = j if j.get('player_ids') else None
        held = mine

    # Past GW1 your squad is locked in, so a from-scratch fifteen is not
    # something you can act on. If we could not read your team, say so loudly
    # rather than quietly printing a different team every day.
    orphaned = (gw > 1 and not args.fresh
                and not (held and len(held.get('player_ids', [])) == 15))

    scenarios, scen_note, transfer = [], '', None
    if held and len(held.get('player_ids', [])) == 15:
        print(f"transfer mode: bank £{held['bank']}m, {held['free_transfers']} free")
        t = optimise.transfers(proj, tot, gws, held['player_ids'],
                               held['bank'], held.get('free_transfers', 1))
        best, hold = t['best'], t['hold']
        names = dict(zip(tot.id, tot.name))
        opts = [dict(label=('Hold — no transfer' if r['transfers'] == 0 else
                            f"{r['transfers']} transfer{'s' if r['transfers']>1 else ''}: "
                            + ', '.join(f"{names.get(o,o)} out" for o in r['out'])),
                     raw=r['raw'], net=r['net'], best=r is best)
                for r in t['options']]
        gain = best['net'] - hold['net']
        verdict = ('Hold. No move clears the cost of making it.' if best is hold else
                   f"Make {best['transfers']} transfer"
                   f"{'s' if best['transfers'] > 1 else ''} — worth "
                   f"{gain:+.1f} points over {len(gws)} gameweeks after hits.")
        ins = [n for n in best['squad'].name if n not in
               set(tot[tot.id.isin(held['player_ids'])].name)]
        detail = (f"Out: {', '.join(names.get(o, str(o)) for o in best['out']) or '—'}. "
                  f"In: {', '.join(ins) or '—'}. "
                  f"Hit cost applied: {best['hit']:.0f}. "
                  "A gain under about 2 points is inside the model's own error — "
                  "when it is that close, holding the transfer is usually worth more "
                  "than the points, because it keeps you flexible for real team news.")
        transfer = dict(verdict=verdict, options=opts, detail=detail)
        result = best
    else:
        print('season-start mode')
        result = optimise.build(proj, tot, gws)
        alts = [('Best available', optimise.build(proj, tot, gws)),
                ('Every player a plausible starter',
                 optimise.build(proj, tot, gws, min_start=0.55))]
        base = alts[0][1]['raw']
        scenarios = [dict(label=l, total=r['raw'], cost=r['cost'])
                     for l, r in alts if r]
        scen_note = ('Requiring all fifteen to be plausible starters costs '
                     f"{abs(alts[1][1]['raw'] - base):.1f} points across the horizon — "
                     'the difference is what cheap bench fodder buys you in the XI.')

    res = result['squad']
    res.to_csv(f'{OUT}/squad.csv', index=False)

    # ---- captain
    nxt = proj[proj.gw == gw]
    cap_pool = nxt[nxt.id.isin(res.id)] if len(res) else nxt
    cap = captain.rank(cap_pool, top=8)
    cap_all = captain.rank(nxt, top=8)
    if cap_all['options'] and cap_all['options'][0]['id'] not in set(res.id):
        cap['note'] += (f" Best in the whole game this week is "
                        f"{cap_all['options'][0]['name']}, who you do not own.")

    xi_ids = list(res[res.start].id)
    cap_id = int(res[res.captain].id.iloc[0]) if res.captain.any() else None
    vc = captain.vice_rank(nxt, xi_ids, cap_id) if cap_id else None

    render(res, proj, tot, ts, fixtures, gws, gw, prov, cap, vc, scenarios,
           scen_note, transfer, result, orphaned, held)
    print(f'\nwrote {DOCS}/index.html')


def render(res, proj, tot, ts, fixtures, gws, gw, prov, cap, vc, scenarios,
           scen_note, transfer, result, orphaned=False, held=None):
    comp = (proj[proj.id.isin(set(res.id))]
            .groupby('id')[['pts_goal', 'pts_ast', 'pts_cs', 'pts_dc',
                            'pts_bonus', 'pts_sv', 'pts_app']].sum())
    squad = []
    for r in res.itertuples():
        pf = proj[proj.id == r.id].sort_values('gw')
        squad.append(dict(
            id=int(r.id), name=r.name, pos=r.pos, team=r.team_name,
            price=float(r.price), p_start=float(r.p_start),
            xp_gw1=round(float(r.xp_next), 2), xp_h=round(float(r.xp_h), 1),
            start=bool(r.start), captain=bool(r.captain),
            vice=bool(vc and r.id == vc['pick_id']),
            owned=float(r.selected_by), prior=bool(r.prior),
            comp={k: round(float(v), 2) for k, v in comp.loc[r.id].items()},
            gws=[dict(gw=int(x.gw), opp=x.opp, home=bool(x.home),
                      xp=round(float(x.xp), 2)) for x in pf.itertuples()]))

    best = {p: [dict(name=r.name, team=r.team_name, price=float(r.price),
                     p_start=round(float(r.p_start), 2),
                     xp_gw1=round(float(r.xp_next), 2), xp_h=round(float(r.xp_h), 1),
                     owned=float(r.selected_by), prior=bool(r.prior),
                     value=round(float(r.value), 2))
                for r in tot[(tot.pos == p) & (tot.status == 'a')]
                .nlargest(15, 'xp_h').itertuples()]
            for p in ('GK', 'DEF', 'MID', 'FWD')}
    dif = tot[(tot.status == 'a') & (tot.selected_by < 8) & (tot.p_start > 0.7)]
    differentials = [dict(name=r.name, pos=r.pos, team=r.team_name,
                          price=float(r.price), xp_h=round(float(r.xp_h), 1),
                          owned=float(r.selected_by))
                     for r in dif.nlargest(12, 'xp_h').itertuples()]

    ev = (prov.get('events') or [{}])[0]
    data = dict(
        squad=squad, ticker=model.ticker(ts, fixtures, gws), best=best,
        differentials=differentials, captain=cap, vice=vc, scenarios=scenarios,
        scenario_note=scen_note, transfer=transfer,
        meta=dict(gw=gw, gw_last=gws[-1], horizon=len(gws),
                  cost=round(float(res.price.sum()), 1),
                  bank=round(100 - float(res.price.sum()), 1)
                  if not transfer else 0.0,
                  xp_gw1=round(float(res[res.start].xp_next.sum()), 1),
                  xp_h=result['raw'],
                  source=prov.get('source', 'unknown'),
                  fetched_at=prov.get('fetched_at'),
                  deadline=prov.get('deadline') or ev.get('deadline_time'),
                  orphaned=bool(orphaned),
                  squad_from_gw=(held or {}).get('from_gw'),
                  free_transfers=(held or {}).get('free_transfers'),
                  stale='mirror' in prov.get('source', '')))

    tpl = open(os.path.join(HERE, 'templates', 'dashboard.html')).read()
    html = tpl.replace('__DATA__', json.dumps(data, default=str))
    open(os.path.join(DOCS, 'index.html'), 'w').write(html)
    json.dump(data, open(os.path.join(OUT, 'dashboard_data.json'), 'w'),
              indent=1, default=str)
    # out/ is committed by the workflow but data/ mostly is not, so drop a copy
    # of the provenance here — when a run goes wrong, which source answered and
    # which gameweek it thought it was are the first two things worth knowing.
    json.dump(dict(prov, gw=gw, horizon=gws, orphaned=bool(orphaned),
                   squad_from_gw=(held or {}).get('from_gw')),
              open(os.path.join(OUT, 'provenance.json'), 'w'), indent=1, default=str)
    write_pool(proj, tot, gws, res)


def write_pool(proj, tot, gws, res):
    """A trimmed, same-origin copy of the candidate pool for the what-if
    explorer. Only the players the optimiser would ever consider, rounded hard —
    this gets fetched on a phone, so it needs to stay small."""
    pool = tot[((tot.xp_h > 3.0) | (tot.price <= 4.5)) & (tot.status != 'u')]
    keep = set(pool.id) | set(res.id)
    p = proj[proj.id.isin(keep)].sort_values(['id', 'gw'])
    players = []
    for pid, g in p.groupby('id', sort=False):
        f = g.iloc[0]
        players.append(dict(
            i=int(pid), n=f['name'], p=f.pos, t=f.team_name,
            c=round(float(f.price), 1), o=float(f.selected_by),
            st=f.status, pr=bool(f.prior),
            ps=[round(float(x), 3) for x in g.p_start],
            pb=[round(float(x), 3) for x in g.p_start_base],
            pm=[round(float(x), 2) for x in g.per_match],
            op=list(g.opp), hm=[bool(x) for x in g.home]))
    json.dump(dict(gws=[int(x) for x in gws], players=players,
                   squad=[int(x) for x in res.id]),
              open(os.path.join(DOCS, 'pool.json'), 'w'), separators=(',', ':'))
    print(f'  pool.json: {len(players)} players, '
          f'{os.path.getsize(os.path.join(DOCS, "pool.json"))//1024} KB')


if __name__ == '__main__':
    main()
