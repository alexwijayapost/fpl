"""Where the data comes from, in order of preference.

1. The official FPL API — live, authoritative. Reachable from GitHub Actions and
   from your own machine, but not from every sandbox.
2. A bootstrap-static.json you saved by hand, dropped next to this repo.
3. The vaastav/Fantasy-Premier-League mirror — always reachable, but a day or
   two behind on injury flags and ownership.

Every fetch records which source answered, and that is printed on the dashboard,
so a stale run is never mistaken for a live one.
"""
import json, os, io, urllib.request, datetime as dt
import pandas as pd

API = 'https://fantasy.premierleague.com/api'
MIRROR = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data'
SEASON, PREV = '2026-27', '2025-26'
UA = {'User-Agent': 'Mozilla/5.0 (compatible; fpl-tools/1.0)'}


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _try(url, timeout=60):
    try:
        return _get(url, timeout)
    except Exception as e:
        print(f'  ! {url.split("/")[2]} unreachable ({type(e).__name__})')
        return None


def load(data_dir, local_json=None):
    """Return (players, fixtures, teams, provenance)."""
    os.makedirs(data_dir, exist_ok=True)
    prov = {'fetched_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}

    raw = _try(f'{API}/bootstrap-static/')
    if raw:
        prov['source'] = 'official FPL API'
    elif local_json and os.path.exists(local_json):
        raw = open(local_json, 'rb').read()
        prov['source'] = f'saved {os.path.basename(local_json)}'
        prov['saved_at'] = dt.datetime.fromtimestamp(
            os.path.getmtime(local_json), dt.timezone.utc).isoformat(timespec='seconds')
    if raw:
        b = json.loads(raw)
        players = pd.DataFrame(b['elements'])
        teams = pd.DataFrame(b['teams'])
        fx = _try(f'{API}/fixtures/')
        fixtures = (pd.DataFrame(json.loads(fx)) if fx else
                    pd.read_csv(io.BytesIO(_get(f'{MIRROR}/{SEASON}/fixtures.csv'))))
        # FPL's own view of where the season is. This is authoritative and must
        # be preferred over inferring the gameweek from fixture 'finished' flags
        # — that inference silently reads a stale preseason file as "nothing has
        # been played yet" and rebuilds the squad from scratch every run.
        prov['events'] = [e for e in b['events'] if e.get('is_next')][:1]
        prov['gw_next'] = next((e['id'] for e in b['events'] if e.get('is_next')), None)
        prov['gw_current'] = next((e['id'] for e in b['events'] if e.get('is_current')), None)
        prov['deadline'] = next((e['deadline_time'] for e in b['events']
                                 if e.get('is_next')), None)
    else:
        prov['source'] = 'vaastav GitHub mirror (may lag on injuries)'
        players = pd.read_csv(io.BytesIO(_get(f'{MIRROR}/{SEASON}/players_raw.csv')),
                              low_memory=False)
        teams = pd.read_csv(io.BytesIO(_get(f'{MIRROR}/{SEASON}/teams.csv')))
        fixtures = pd.read_csv(io.BytesIO(_get(f'{MIRROR}/{SEASON}/fixtures.csv')))

    for df, name in ((players, 'players_raw'), (teams, 'teams'), (fixtures, 'fixtures')):
        df.to_csv(os.path.join(data_dir, f'{name}.csv'), index=False)

    # history: always from the mirror, and it never changes once a season ends
    for season, tag in ((PREV, 'prev'),):
        for f in ('players_raw', 'teams'):
            p = os.path.join(data_dir, f'{tag}_{f}.csv')
            if not os.path.exists(p):
                open(p, 'wb').write(_get(f'{MIRROR}/{season}/{f}.csv'))
        p = os.path.join(data_dir, f'{tag}_merged_gw.csv')
        if not os.path.exists(p):
            open(p, 'wb').write(_get(f'{MIRROR}/{season}/gws/merged_gw.csv', 180))

    # this season's per-gameweek data, once it exists
    cur = _try(f'{MIRROR}/{SEASON}/gws/merged_gw.csv', 180)
    if cur:
        open(os.path.join(data_dir, 'merged_gw.csv'), 'wb').write(cur)

    snapshot(players, data_dir)
    json.dump(prov, open(os.path.join(data_dir, 'provenance.json'), 'w'), indent=1,
              default=str)
    print(f"  source: {prov['source']}")
    return players, fixtures, teams, prov


def snapshot(players, data_dir):
    """Daily price and ownership snapshot. This history cannot be recovered
    later from anywhere, so it has to be collected from day one."""
    d = os.path.join(data_dir, 'snapshots')
    os.makedirs(d, exist_ok=True)
    cols = [c for c in ['id', 'web_name', 'now_cost', 'selected_by_percent',
                        'transfers_in_event', 'transfers_out_event', 'status',
                        'chance_of_playing_next_round'] if c in players.columns]
    players[cols].to_csv(
        os.path.join(d, f'{dt.date.today().isoformat()}.csv'), index=False)


def my_team(entry_id, gw, free_transfers=1):
    """Your current 15 from the FPL API.

    `gw` is the gameweek to read picks from — the last one that has started.
    Walks backwards a few gameweeks so a single missing week (an API blip, or a
    week you had no team) does not silently drop us back to season-start mode.
    """
    if not entry_id or not gw or gw < 1:
        return None
    for g in range(gw, max(gw - 4, 0), -1):
        raw = _try(f'{API}/entry/{entry_id}/event/{g}/picks/')
        if not raw:
            continue
        d = json.loads(raw)
        picks = [p['element'] for p in d.get('picks', [])]
        if len(picks) != 15:
            continue
        e = _try(f'{API}/entry/{entry_id}/')
        bank = json.loads(e)['last_deadline_bank'] / 10.0 if e else 0.0
        # The API does not expose remaining free transfers, so this is a config
        # value rather than a guess. entry_history.event_transfers is transfers
        # *made* that week, which is a different thing entirely.
        return dict(player_ids=picks, bank=bank, from_gw=g,
                    free_transfers=free_transfers)
    return None


def league(kind, league_id):
    """kind: 'classic' or 'h2h'."""
    raw = _try(f'{API}/leagues-{kind}/{league_id}/standings/')
    return json.loads(raw) if raw else None
