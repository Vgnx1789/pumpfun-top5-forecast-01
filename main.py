"""pumpfun-top5-forecast-01 entry: a Monte-Carlo forecaster for the top 5
pump.fun coins by market cap at the answer reads.

Reads $MAKEFUN_INPUT_DIR (default /input): task.json + history.ndjson.
Writes $MAKEFUN_OUTPUT_DIR (default /output)/forecast.json.

Method: for every coin the history ranks, fit its log10-cap (in SOL)
trajectory across the marks; extrapolate to the mean answer-read time with
a shrunk momentum term and a staleness decay; then simulate the ranking
many times (fixed-seed RNG, so the run is byte-for-byte deterministic),
injecting synthetic post-cut newcomers so part of every place's mass
honestly stays unnamed. Place probabilities are the simulation
frequencies; each place's size distribution is the simulated histogram
of that place's cap, lightly smoothed onto the task's bin grid.
Standard library only, no network.
"""
import json
import math
import os
from bisect import bisect_right
from pathlib import Path

# --- knobs (round numbers; tuned on walk-forward rehearsals) ---
MOMENTUM = 0.5        # weight on a coin's shrunk log-cap slope (per hour)
SIGMA0 = 1.0          # log10-cap diffusion per sqrt(hour) of horizon
CURVE_EXTRA_SIGMA = 0.15
STALE_DECAY = 0.10    # log10 lost per hour since the coin was last ranked
NEW_PER_HOUR = 18.0   # synthetic post-cut newcomers per hour of read span
NEW_MU_DROP = 0.3     # newcomers centre this far below the top coin's mean
NEW_SIGMA = 0.9
NEW_WINDOW_H = 2.0    # only newcomers born within ~2h before a read contend
                      # at it, so the synthetic pool caps at this window
DRIFT_TAU_H = 3.0     # momentum extrapolation saturates: H*tau/(H+tau)
TRIALS = 3000
MAX_CANDIDATES = 250
MIN_NAMED_P = 0.004
SIZE_NAMED_MASS = 0.99

MASK64 = (1 << 64) - 1


class Rng:
    """SplitMix64 + Box-Muller: deterministic across runs and versions."""

    def __init__(self, seed):
        self.state = seed & MASK64
        self.spare = None

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self):
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def normal(self):
        if self.spare is not None:
            v, self.spare = self.spare, None
            return v
        u1 = 1.0 - self.uniform()
        u2 = self.uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        self.spare = r * math.sin(2.0 * math.pi * u2)
        return r * math.cos(2.0 * math.pi * u2)


def parse_time(s):
    # ISO 8601 UTC like 2026-11-11T20:00:00.000Z -> epoch seconds (no datetime
    # needed: use a civil-date conversion).
    date, _, rest = s.partition('T')
    y, mo, d = (int(x) for x in date.split('-'))
    hh, mm, ss = rest.rstrip('Z').split(':')
    days = days_from_civil(y, mo, d)
    return days * 86400 + int(hh) * 3600 + int(mm) * 60 + float(ss)


def days_from_civil(y, m, d):
    y -= 1 if m <= 2 else 0
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m - 3 if m > 2 else m + 9) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def size_of(lamports):
    return math.log10(int(lamports)) - 9.0


def floor6(x):
    return math.floor(x * 1_000_000) / 1_000_000


def build(input_dir):
    task = json.loads(Path(input_dir, 'task.json').read_text())
    places = task['places']
    edges = task['size']['edges']
    history_until = parse_time(task['history_until'])
    read_times = [parse_time(r) for r in task['reads']]
    h_mean = max(0.05, (sum(read_times) / len(read_times) - history_until) / 3600.0)
    read_span = max(0.05, (read_times[-1] - history_until) / 3600.0)

    marks = []
    for line in Path(input_dir, 'history.ndjson').read_text().split('\n'):
        if not line:
            continue
        m = json.loads(line)
        if m.get('ok') is True and isinstance(m.get('rows'), list) and m['rows']:
            marks.append(m)
    if not marks:
        raise ValueError('the history holds no ok mark with rows')

    # Per-coin trajectories: mint -> list of (t_hours since first mark, size),
    # plus the last row's static fields.
    t_ref = parse_time(marks[0]['at'])
    series = {}
    last_row = {}
    for m in marks:
        t = (parse_time(m['at']) - t_ref) / 3600.0
        for row in m['rows']:
            mint = row['mint']
            series.setdefault(mint, []).append((t, size_of(row['market_cap_lamports'])))
            last_row[mint] = row
    cut_t = (history_until - t_ref) / 3600.0
    last_mark_t = (parse_time(marks[-1]['at']) - t_ref) / 3600.0

    cands = []
    top_size = max(s[-1][1] for s in series.values())
    for mint, pts in series.items():
        pts.sort()
        last_t, last_size = pts[-1]
        n = len(pts)
        if n >= 2:
            t0_, s0_ = pts[0]
            span = max(last_t - t0_, 1e-9)
            # OLS slope of size on time
            mt = sum(p[0] for p in pts) / n
            ms = sum(p[1] for p in pts) / n
            num = sum((p[0] - mt) * (p[1] - ms) for p in pts)
            den = sum((p[0] - mt) ** 2 for p in pts)
            slope = num / den if den > 0 else (s0_ and (last_size - s0_) / span) or 0.0
            slope *= n / (n + 2.0)  # shrinkage
        else:
            slope = 0.0
        slope = max(-2.0, min(2.0, slope))
        stale_h = max(0.0, last_mark_t - last_t)
        h_drift = h_mean * DRIFT_TAU_H / (h_mean + DRIFT_TAU_H)
        mu = last_size + MOMENTUM * slope * h_drift - STALE_DECAY * stale_h
        row = last_row[mint]
        sigma = SIGMA0 * math.sqrt(h_mean) + (CURVE_EXTRA_SIGMA if row.get('venue') == 'curve' else 0.0)
        if last_size >= top_size - 3.0 or mu >= top_size - 2.5:
            cands.append((mint, mu, sigma))
    cands.sort(key=lambda c: -c[1])
    cands = cands[:MAX_CANDIDATES]
    if len(cands) < places:
        raise ValueError('fewer candidate coins than places')

    n_new = max(1, int(round(NEW_PER_HOUR * min(read_span, NEW_WINDOW_H))))
    top_mu = cands[0][1]
    new_mu = top_mu - NEW_MU_DROP

    rng = Rng(20261112)
    place_hits = [dict() for _ in range(places)]
    place_sizes = [[] for _ in range(places)]
    mus = [c[1] for c in cands]
    sigs = [c[2] for c in cands]
    mints = [c[0] for c in cands]
    for _ in range(TRIALS):
        vals = [mus[i] + sigs[i] * rng.normal() for i in range(len(cands))]
        vals.extend(new_mu + NEW_SIGMA * rng.normal() for _ in range(n_new))
        order = sorted(range(len(vals)), key=lambda i: -vals[i])
        for k in range(places):
            idx = order[k]
            place_sizes[k].append(vals[idx])
            if idx < len(cands):
                d = place_hits[k]
                d[mints[idx]] = d.get(mints[idx], 0) + 1

    probabilities = []
    exp_score = {}
    for k in range(places):
        rows = []
        for mint, cnt in place_hits[k].items():
            p = cnt / TRIALS
            if p >= MIN_NAMED_P:
                rows.append((mint, p))
                exp_score[mint] = exp_score.get(mint, 0.0) + p * (places - k)
        rows.sort(key=lambda r: (-r[1], r[0]))
        for mint, p in rows[:60]:
            fp = floor6(p)
            if fp > 0:
                probabilities.append({"mint": mint, "place": k + 1, "p": fp})
    probabilities.sort(key=lambda r: (r['place'], -r['p'], r['mint']))
    top5 = [m for m, _ in sorted(exp_score.items(), key=lambda kv: (-kv[1], kv[0]))[:places]]

    nbins = len(edges) + 1
    sizes = []
    for k in range(places):
        hist = [0.0] * nbins
        for v in place_sizes[k]:
            hist[bisect_right(edges, v)] += 1.0
        tot = sum(hist) or 1.0
        smooth = []
        for b in range(nbins):
            c = hist[b]
            lft = hist[b - 1] if b > 0 else 0.0
            rgt = hist[b + 1] if b < nbins - 1 else 0.0
            smooth.append((0.7 * c + 0.15 * lft + 0.15 * rgt) / tot)
        sm = sum(smooth) or 1.0
        for b in range(nbins):
            fp = floor6(smooth[b] / sm * SIZE_NAMED_MASS)
            if fp >= 0.0005:
                sizes.append({"place": k + 1, "bin": b, "p": fp})
    sizes.sort(key=lambda r: (r['place'], r['bin']))
    return {"probabilities": probabilities, "top5": top5, "sizes": sizes}


def main():
    inp = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    outp = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    fc = build(inp)
    Path(outp).mkdir(parents=True, exist_ok=True)
    task = json.loads(Path(inp, 'task.json').read_text())
    Path(outp, task['output']).write_text(json.dumps(fc) + '\n')


if __name__ == '__main__':
    main()
