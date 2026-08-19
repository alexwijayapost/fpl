"""Team strength and per-player expected points.

The design follows one principle: spend the modelling effort on *minutes*, not on
a fancier goals model. Public models already match commercial ones on predicting
high returns; where they lose is predicting blanks, and blanks are a minutes
problem. So start probability gets the careful treatment here, and the scoring
components stay deliberately simple and inspectable.
"""
import numpy as np, pandas as pd
from scipy.stats import poisson

HOME_ADV = 1.10
AWAY_ADV = 1 / HOME_ADV
LEAGUE_GPG = 1.384                       # goals per team per game, 2025-26

POS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
GOAL_PTS = {'GK': 10, 'DEF': 6, 'MID': 5, 'FWD': 4}
CS_PTS = {'GK': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}
DEFCON_THR = {'GK': 99, 'DEF': 10, 'MID': 12, 'FWD': 12}
EXPECTED_STARTERS = {'GK': 1.0, 'DEF': 4.2, 'MID': 4.2, 'FWD': 1.6}
PROMOTED_ATT, PROMOTED_DEF = 0.80, 1.30  # first-season prior for a promoted side


def _ratings(gw, teams):
    """Attack and defence multipliers from per-match expected goals."""
    if gw is None or len(gw) == 0:
        return None
    id2n = dict(zip(teams.id, teams.name))
    g = gw[gw.minutes > 0].copy()
    g['opp_name'] = g.opponent_team.map(id2n)
    xgf = g.groupby(['team', 'fixture']).agg(
        xg=('expected_goals', 'sum'), gf=('goals_scored', 'sum'),
        opp=('opp_name', 'first')).reset_index()
    xga = (g[g.minutes >= 85].groupby(['team', 'fixture'])['expected_goals_conceded']
           .max().reset_index().rename(columns={'expected_goals_conceded': 'xgc'}))
    tm = xgf.merge(xga, on=['team', 'fixture'], how='left')
    lookup = dict(zip(zip(tm.team, tm.fixture), tm.gf))
    tm['ga'] = [lookup.get((r.opp, r.fixture), np.nan) for r in tm.itertuples()]
    a = tm.groupby('team').agg(games=('fixture', 'nunique'), xgf=('xg', 'mean'),
                               xgc=('xgc', 'mean'), gf=('gf', 'mean'),
                               ga=('ga', 'mean')).reset_index()
    # expected goals are steadier than goals; goals carry the finishing signal
    a['att_raw'] = 0.7 * a.xgf + 0.3 * a.gf
    a['def_raw'] = 0.7 * a.xgc + 0.3 * a.ga
    K = 8.0                                          # shrink toward league average
    la, ld = a.att_raw.mean(), a.def_raw.mean()
    a['att'] = (a.att_raw * a.games + la * K) / (a.games + K) / la
    a['dfn'] = (a.def_raw * a.games + ld * K) / (a.games + K) / ld
    return a.rename(columns={'team': 'name'})[['name', 'att', 'dfn', 'games']]


def team_strength(teams, prev_gw, prev_teams, cur_gw=None):
    prev = _ratings(prev_gw, prev_teams)
    out = teams[['id', 'name', 'short_name']].merge(prev, on='name', how='left')
    cur = _ratings(cur_gw, teams) if cur_gw is not None else None
    if cur is not None and len(cur) and cur.games.max() >= 3:
        n = float(cur.games.max())
        w = min(n / 10.0, 0.70)                      # weight this season more each week
        out = out.merge(cur[['name', 'att', 'dfn']], on='name', how='left',
                        suffixes=('', '_c'))
        for c in ('att', 'dfn'):
            out[c] = np.where(out[f'{c}_c'].notna(),
                              (1 - w) * out[c].fillna(out[f'{c}_c']) + w * out[f'{c}_c'],
                              out[c])
        out.attrs['blend'] = w
    out['promoted'] = out.att.isna()
    out.loc[out.promoted, 'att'] = PROMOTED_ATT
    out.loc[out.promoted, 'dfn'] = PROMOTED_DEF
    return out


def start_probability(cur, cur_gw=None):
    """P(start). The weakest link in any public model — and the thing to override
    by hand when press-conference team news lands."""
    if cur_gw is not None and len(cur_gw):
        played = max(int(cur_gw.GW.max()), 1)
        starts = cur_gw.groupby('element').starts.sum()
        mins = cur_gw.groupby('element').minutes.sum()
        cur['hist_start'] = (cur.id.map(starts) / played).clip(0, 1)
        cur.loc[cur.id.map(mins).fillna(0) < 180, 'hist_start'] = np.nan
        cur['no_history'] = cur.id.map(mins).fillna(0) < 180
    else:
        cur['hist_start'] = (cur.starts / 38.0).clip(0, 1)
        cur.loc[cur.minutes < 200, 'hist_start'] = np.nan
        cur['no_history'] = cur.minutes < 400

    def rank_in_club(g):
        n = EXPECTED_STARTERS[g.name[1]]
        r = g.price.rank(ascending=False, method='first')
        return 1 / (1 + np.exp((r - n - 0.5) / 0.9))

    role_club = cur.groupby(['team', 'pos'], group_keys=False).apply(
        rank_in_club, include_groups=False)
    # Rank inside a club says nothing where the whole squad sits at the price
    # floor — which is every promoted side. FPL's absolute price for the position
    # is the better signal there: a 4.5 forward is priced as a non-starter.
    role_abs = cur.groupby('pos').price.transform(lambda s: s.rank(pct=True)) ** 0.8
    cur['role'] = 0.5 * role_club + 0.5 * role_abs
    cur['p_start'] = np.where(cur.hist_start.notna(),
                              0.55 * cur.hist_start.fillna(0) + 0.45 * cur.role,
                              cur.role)
    # nothing to learn from -> the number is a guess, so discount it rather than
    # letting the optimiser spend real money on it
    cur.loc[cur.no_history, 'p_start'] *= 0.80

    ch = pd.to_numeric(cur.get('chance_of_playing_next_round'), errors='coerce')
    cur['avail'] = np.where(ch.notna(), ch / 100.0, 1.0)
    cur.loc[cur.status.isin(['i', 's', 'u', 'n']), 'avail'] = 0.0
    cur['p_start_base'] = cur.p_start.clip(0, 0.97)
    cur['p_start'] = (cur.p_start * cur.avail).clip(0, 0.97)
    return cur


def prepare(players, teams, prev_players, prev_teams, ts):
    cur = players.copy()
    cur['pos'] = cur.element_type.map(POS)
    cur['price'] = cur.now_cost / 10.0
    for c in ('expected_goals', 'expected_assists', 'defensive_contribution',
              'saves', 'bonus', 'yellow_cards', 'minutes', 'starts',
              'selected_by_percent', 'penalties_order'):
        if c in cur:
            cur[c] = pd.to_numeric(cur[c], errors='coerce')

    att = dict(zip(ts.id, ts.att))
    n_prev = dict(zip(prev_teams.id, prev_teams.name))
    ts_n = ts.set_index('name')
    old = {c: ts_n.att.get(n_prev.get(t), 1.0)
           for c, t in zip(prev_players.code, prev_players.team)}
    cur['old_att'] = cur.code.map(old).fillna(1.0)
    cur['new_att'] = cur.team.map(att)
    cur['new_dfn'] = cur.team.map(dict(zip(ts.id, ts.dfn)))

    m90 = cur.minutes.clip(lower=1) / 90.0
    ok = cur.minutes >= 400
    for src, dst in [('expected_goals', 'xg90'), ('expected_assists', 'xa90'),
                     ('bonus', 'bonus90'), ('saves', 'saves90'),
                     ('yellow_cards', 'yc90'), ('defensive_contribution', 'dc90')]:
        cur[dst] = (cur[src] / m90).where(ok)
    cur['pband'] = cur.groupby('pos').price.transform(
        lambda s: pd.qcut(s.rank(method='first'), 5, labels=False))
    for c in ('xg90', 'xa90', 'bonus90', 'saves90', 'yc90', 'dc90'):
        cur[c] = (cur[c].fillna(cur.groupby(['pos', 'pband'])[c].transform('median'))
                  .fillna(cur.groupby('pos')[c].transform('median')).fillna(0))
    cur['pen1'] = cur.penalties_order == 1
    # small, because last season's xG already embeds penalties for existing takers
    cur.loc[cur.pen1, 'xg90'] += 0.05
    return cur


def project(cur, ts, fixtures, gws):
    att, dfn = dict(zip(ts.id, ts.att)), dict(zip(ts.id, ts.dfn))
    short = dict(zip(ts.id, ts.short_name))
    fx = fixtures[fixtures.event.isin(gws)]
    sched = pd.DataFrame(
        [(int(f.event), int(a), int(b), h) for _, f in fx.iterrows()
         for a, b, h in ((f.team_h, f.team_a, True), (f.team_a, f.team_h, False))],
        columns=['gw', 'team', 'opp', 'home'])

    rows = []
    for p in cur.itertuples():
        for f in sched[sched.team == p.team].itertuples():
            # a flagged player is discounted hard for the next two gameweeks,
            # then allowed most of the way back to his baseline role
            i = gws.index(f.gw)
            ps = p.p_start if i < 2 else p.p_start + (p.p_start_base - p.p_start) * 0.6
            ha = HOME_ADV if f.home else AWAY_ADV
            odef, oatt = dfn[f.opp], att[f.opp]
            shift = float(np.clip(p.new_att / p.old_att if p.old_att else 1, 0.6, 1.7))
            mult = shift * odef * ha
            xgc = LEAGUE_GPG * oatt * p.new_dfn * (AWAY_ADV if f.home else HOME_ADV)
            p_cs = float(poisson.pmf(0, xgc))

            g = p.xg90 * mult * GOAL_PTS[p.pos]
            a = p.xa90 * mult * 3.0
            cs = p_cs * CS_PTS[p.pos]
            gc = (-sum(poisson.pmf(k, xgc) * (k // 2) for k in range(9))
                  if p.pos in ('GK', 'DEF') else 0.0)
            sv = (p.saves90 * oatt) / 3.0 if p.pos == 'GK' else 0.0
            # DefCon is a threshold, not an average: what scores is P(hitting 10
            # or 12 actions in a match). No opponent adjustment — 2025-26 shows
            # no reliable relationship between opponent strength and hit rate.
            thr = DEFCON_THR[p.pos]
            dc = (2.0 * (1 - float(poisson.cdf(thr - 1, p.dc90)))
                  if thr < 99 and p.dc90 > 0 else 0.0)
            per = g + a + cs + gc + sv + dc + p.bonus90 - p.yc90

            rows.append(dict(
                id=p.id, name=p.web_name, pos=p.pos, team=p.team,
                team_name=short[p.team], price=p.price, gw=int(f.gw),
                opp=short[f.opp], home=bool(f.home), p_start=round(ps, 3),
                xp=ps * per + ps * 1.9, pts_goal=ps * g, pts_ast=ps * a,
                pts_cs=ps * cs, pts_dc=ps * dc, pts_bonus=ps * p.bonus90,
                pts_sv=ps * sv, pts_app=ps * 1.9, p_cs=p_cs,
                xg=p.xg90 * mult, xa=p.xa90 * mult, bonus90=p.bonus90,
                selected_by=p.selected_by_percent, status=p.status,
                news=p.news, pen1=bool(p.pen1), prior=bool(p.no_history)))

    proj = pd.DataFrame(rows)
    tot = (proj.groupby(['id', 'name', 'pos', 'team_name', 'price', 'selected_by',
                         'status', 'pen1', 'prior'])
           .agg(xp_h=('xp', 'sum'), xp_next=('xp', lambda s: s.iloc[0]),
                p_start=('p_start', 'mean'), games=('gw', 'count')).reset_index())
    tot['value'] = tot.xp_h / tot.price
    return proj, tot


def ticker(ts, fixtures, gws):
    att, dfn = dict(zip(ts.id, ts.att)), dict(zip(ts.id, ts.dfn))
    short = dict(zip(ts.id, ts.short_name))
    rows = []
    for _, f in fixtures[fixtures.event.isin(gws)].iterrows():
        for t, o, h in ((f.team_h, f.team_a, True), (f.team_a, f.team_h, False)):
            raw = (0.5 * att[o] + 0.5 * (2 - dfn[o])) * (0.94 if h else 1.06)
            rows.append(dict(gw=int(f.event), team=short[t], opp=short[o],
                             home=h, raw=raw))
    tk = pd.DataFrame(rows)
    lo, hi = tk.raw.quantile(0.03), tk.raw.quantile(0.97)
    tk['fdr'] = ((tk.raw - lo) / (hi - lo) * 4 + 1).clip(1, 5).round(2)
    return {t: dict(fixtures=[dict(gw=int(r.gw), opp=r.opp, home=bool(r.home),
                                   fdr=float(r.fdr))
                              for r in g.sort_values('gw').itertuples()],
                    avg=round(float(g.fdr.mean()), 2))
            for t, g in tk.groupby('team')}
