# FPL 2026/27 — automatic weekly recommendation

Every day this repository pulls the live Fantasy Premier League data, re-runs the
model, and rebuilds a dashboard you can open on your phone. Twice a week, ahead
of the deadline, it also works out whether you should make a transfer.

You do not have to run anything. Once it's set up, you open one link.

---

## Setup — about 10 minutes, once

You need a free GitHub account. Nothing else, and no software on your Mac.

### 1. Create the account

Go to **github.com** and sign up. Any username. Verify the email.

### 2. Create the repository

1. Click the **+** in the top right → **New repository**.
2. Repository name: `fpl` (or anything you like).
3. Choose **Private**. Your team is your business.
4. Leave everything else alone. Click **Create repository**.

### 3. Upload these files

On the new empty repository page, click **uploading an existing file**.

Drag the entire contents of the `fpl-tools` folder in. GitHub keeps the folder
structure, so `.github/workflows/fpl.yml`, the `fpl/` folder and `data/` all land
in the right place.

> If GitHub refuses the `.github` folder because it starts with a dot, upload
> everything else first, then use **Add file → Create new file**, type
> `.github/workflows/fpl.yml` as the filename, and paste the contents of that
> file in.

Scroll down, click **Commit changes**.

### 4. Let it write back to itself

**Settings** (tab along the top) → **Actions** → **General** → scroll to
**Workflow permissions** → select **Read and write permissions** → **Save**.

This is what lets the daily run save its own data snapshots.

### 5. Turn on the web page

**Settings** → **Pages** → under **Source** choose **GitHub Actions**.

### 6. Run it once by hand

**Actions** tab → **FPL weekly recommendation** in the left sidebar →
**Run workflow** button → **Run workflow**.

Give it two minutes. When the tick goes green, your dashboard is live at:

```
https://<your-username>.github.io/fpl/
```

Bookmark that on your phone home screen. That's the whole thing.

---

## What happens from then on

| When | What runs |
|---|---|
| Every day, 07:00 UTC | Pulls prices and ownership, saves a dated snapshot, rebuilds the dashboard |
| Thursday & Friday, 09:00 UTC | Full pre-deadline run — transfer recommendation and captain call |
| Whenever you press the button | Same as above, on demand |

The daily snapshot matters more than it looks. Price-change and ownership history
exists nowhere on the internet — nobody sells it to you. It only exists if you
start recording it, so the earlier this starts running, the more the price model
has to work with later in the season.

---

## Changing things

Everything you'd want to adjust lives in `config.json`:

```json
{
  "entry_id": 5951977,
  "leagues": { "classic": 1018282, "h2h": 1018283 },
  "horizon": 6
}
```

- `entry_id` — your FPL team. The run reads your actual squad from this, so you
  never have to tell it what you own.
- `horizon` — how many gameweeks ahead to plan. Six is a reasonable default;
  raising it makes the model more patient about fixture swings.

To change the schedule, edit the `cron:` lines in `.github/workflows/fpl.yml`.
They're in UTC — Jakarta is UTC+7, so `0 9 * * 5` is Friday 4pm your time.

---

## Reading the dashboard

**The transfer call** is the part to actually act on. It shows what holding is
worth against what each move is worth *after* the −4 hits, so the recommendation
is a number, not a hunch. If a move gains less than about 2 points over the
horizon, hold — that's inside the model's own error, and a saved transfer is
worth real money when actual team news lands on Friday.

**Captain** gives two answers on purpose. The steady pick maximises expected
points, which is what your classic league rewards. The ceiling pick has a better
chance of a double-digit haul on a player fewer rivals own, which is what wins
head-to-head weeks. When they disagree, the dashboard says so.

**Start%** is the number to distrust first. It's built from last season's starts
and the player's price, and it cannot know what a manager said in a press
conference on Friday morning. When you see a projection that looks wrong, this is
almost always why.

Anyone tagged **guess** has no Premier League minutes to model from — usually a
promoted-club player or a new signing. The projection is a price-based prior, not
evidence.

---

## What's in here

```
run.py                     the weekly run
config.json                your team and league ids
fpl/sources.py             where data comes from, and how stale it is
fpl/model.py               team strength, start probability, expected points
fpl/optimise.py            squad selection and the transfer/hit decision
fpl/captain.py             points distributions — ceiling, blank risk
templates/dashboard.html   the page
data/snapshots/            daily price + ownership history, accumulating
docs/index.html            the published dashboard
```

## Running it on your own Mac instead

```
pip install -r requirements.txt
python3 run.py
open docs/index.html
```

## Known limits

- Start probability is the weak link, here and in every public model. The gap
  between free models and paid services is almost entirely expected-minutes
  forecasting, which is an information problem — press conferences and beat
  reporters — not a modelling one.
- The projection assumes each fixture is independent. It doesn't model
  rotation ahead of a European tie, or a manager resting players with a cup game
  three days later.
- Promoted clubs carry a flat first-season prior until they've played enough
  matches to rate on their own. Expect their players to be mispriced by the model
  for the first month.
- Blank and double gameweeks aren't detected yet — that needs the FA Cup and
  European calendars, and it matters from about December.
