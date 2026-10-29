"""
Preprocesses raw CGWB groundwater data into one daily series per district.

v4 rebuild (2026-10). The v3 district series had 376 day-to-day jumps > 2 m, and
81-84% of them fell on days when the SET OF REPORTING WELLS changed — the district
value was the median of whichever wells reported that day, so a well dropping out
moved the "water table". Audit findings this version fixes:

  * a telemetry placeholder (exact +1.000) — e.g. Akola CGWBNG0501 sent it for 24
    days — pushed 6 good wells under the old 95%-one-sign rule and the whole station
    was dropped; 12 telemetric wells were lost that way (Nashik kept 2 of 5)
  * the positive readings of mixed-sign stations are fault periods, not a different
    sign convention (Bhandara reads +20..50 m against a real level of -3.4 m), so
    they are dropped as readings, never as whole stations
  * 484 runs of >= 8 identical readings (stuck loggers), and dry-well plateaus where
    the logger sits at sensor depth (Dhule CGWNAG2183 pinned at -54.55 m)
  * the Hampel window was 7 READINGS (~1.75 days at ~4 readings/day), so any fault
    lasting a day or more became the local median and passed; MAD = 0 runs passed
  * the 5 m/day district cap compared each day to the previous UNCAPPED value, so it
    removed only the first day of a multi-day fault (and also the good day after a
    one-day spike)
  * bfill(limit=7) filled unobserved days with FUTURE readings

Pipeline
  1. raw readings : telemetric panel only, |v| <= 100 m, drop exact 0 and +1.000
  2. sign         : per-station majority sign; drop minority-sign readings and the
                    majority readings within +-SIGN_GUARD_DAYS of them; depth -> -|v|
  3. stuck runs   : >= STUCK_RUN identical consecutive raw readings -> dropped
  4. station-day means, then per station:
       two-sided time-based Hampel (a spike must differ from BOTH the previous and the
       next week, so a genuine step change is never mistaken for a spike);
       plateau / censoring check (30-day IQR ~ 0 in a well that normally moves);
       dead sensors and stations with < MIN_GOOD_DAYS good days are dropped
  5. step detection against the LAST ACCEPTED value: a large jump is kept if the
     district's other wells moved the same way; if instead the new level persists it
     is a REGIME BREAK and the station is split into a new pseudo-station segment
     (its own offset); otherwise the value is rejected
  6. drift filter per segment (monotonic equilibration of new boreholes)
  7. district aggregation with a rank-1 factor model: every segment p is modelled as
         x_p(t) = mu_p + beta_p * D(t)
     (own offset, and an amplitude shrunk toward 1), and D(t) is the MEDIAN over the
     segments reporting on day t of (x_p - mu_p) / beta_p. Offsets are estimated jointly
     with D, so a well that joins or leaves no longer moves the series.
     Method chosen by audit_groundwater_target.py (jumps > 2 m 329 -> 193, best
     leave-one-station-out correlation) — see AGG_METHOD.
  8. gwl_observed = 1 only if enough of the active panel really reported that day;
     district-level jumps are flagged against the last ACCEPTED value. No fills.

Input  : data/raw/groundwater_maharashtra_23_25_uncleaned.csv
Output : data/clean/groundwater_clean.csv
         data/clean/groundwater_station_daily.csv   (station/segment daily, QC'd)
         data/clean/station_qc_log.csv              (what was removed and why)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / "data" / "raw" / "groundwater_maharashtra_23_25_uncleaned.csv"
CLEAN = HERE.parent / "data" / "clean"
OUTPUT = CLEAN / "groundwater_clean.csv"

START = "2023-01-01"
END = "2026-06-04"   # master calendar end = last real CGWB reading (re-checked 2026-10-02: WRIS
# returns nothing later for any district/agency). The covariate files (rain/ET/soil/temp/sentinel)
# run to 2026-10-01 but merge_master.py left-joins onto this calendar, so they are trimmed to it.

# ─── thresholds ───────────────────────────────────────────────────────────────
MAX_ABS_DEPTH = 100.0       # |value| above this is a sensor fault (raw max +511 m)
PLACEHOLDERS = [1.0]        # telemetry placeholder values (2,709 readings of exactly +1.000)
SIGN_GUARD_DAYS = 2         # majority-sign readings this close to a fault reading are dropped too
STUCK_RUN = 12              # identical consecutive raw readings (~3 days at 4/day) = stuck logger
HAMPEL_HALF = 7             # days on each side
HAMPEL_MIN_PTS = 4
HAMPEL_K = 4.0
PLATEAU_WIN = 30            # days
PLATEAU_IQR = 0.03          # m: a 30-day window this flat ...
PLATEAU_NORMAL_IQR = 0.30   # m: ... in a well whose typical 30-day IQR exceeds this
DEAD_IQR = 0.05             # m: lifetime IQR below this = dead sensor
MIN_GOOD_DAYS = 180         # station must keep this many good days
STEP_MIN = 3.0              # m: smallest jump that is ever questioned
STEP_SIGMA_K = 8.0          # ... or this many robust daily-change SDs (x sqrt(gap))
COMOVE_FRAC = 0.30          # other wells moved >= this fraction of the jump, same sign -> real
LOOKAHEAD = 10              # days used to decide transient fault vs persistent new level
PERSIST_FRAC = 0.70
OUTAGE_DAYS = 7             # a persistent jump after a gap this long is a recalibration
MIN_SEGMENT_DAYS = 30
DRIFT_R2, DRIFT_RANGE = 0.60, 20.0
BETA_CLIP = (0.25, 3.0)
FACTOR_ITERS = 25
OBS_FRAC = 0.34             # real reporting segments >= max(1, ceil(OBS_FRAC * active))
DISTRICT_JUMP_MIN = 3.0
DISTRICT_JUMP_K = 6.0


# ─── helpers ──────────────────────────────────────────────────────────────────
def robust_daily_sd(s: pd.Series) -> float:
    """Robust SD of consecutive-day changes of a calendar-indexed daily series."""
    d = s.diff().dropna()
    if len(d) < 10:
        return 0.05
    return float(max(1.4826 * np.median(np.abs(d - np.median(d))), 0.005))


def hampel_two_sided(s: pd.Series, sigma_d: float) -> pd.Series:
    """True where a value is a spike: far from BOTH the previous and the next week.

    A genuine step (monsoon recharge, a pump switching off) sits close to one side,
    so it is never flagged; a fault of a few days differs from both and is.
    """
    left = s.shift(1).rolling(HAMPEL_HALF, min_periods=HAMPEL_MIN_PTS).median()
    right = s[::-1].shift(1).rolling(HAMPEL_HALF, min_periods=HAMPEL_MIN_PTS).median()[::-1]
    centre = s.rolling(2 * HAMPEL_HALF + 1, center=True, min_periods=5).median()
    mad = (s - centre).abs().rolling(2 * HAMPEL_HALF + 1, center=True, min_periods=5).median()
    thr = HAMPEL_K * np.maximum(np.maximum(1.4826 * mad.fillna(0), 3 * sigma_d), 0.02)
    # A spike lies OUTSIDE the band spanned by the previous-week and next-week medians, on
    # the same side of both. A point on a steep monsoon rise lies BETWEEN them (below next
    # week, above last week) and must never be flagged — a "far from both" test without the
    # same-side condition removed real recharge ramps (Raigad, Beed) in a first draft.
    hi_ = np.maximum(left, right)
    lo_ = np.minimum(left, right)
    both = left.notna() & right.notna()
    out_both = both & (((s - hi_) > thr) | ((lo_ - s) > thr))
    one = left.notna() ^ right.notna()
    side = left.where(left.notna(), right)
    out_one = one & ((s - side).abs() > thr) & ((s - centre).abs() > thr)
    return s.notna() & (out_both | out_one)


def plateau_mask(s: pd.Series) -> pd.Series:
    """True on days inside an implausibly flat window of a well that normally moves.

    Two tests, both only for wells whose typical 30-day IQR exceeds PLATEAU_NORMAL_IQR:
      * a 30-day window with IQR < PLATEAU_IQR (stuck logger with jitter)
      * a 60-day window whose full range is < min(0.5 m, 5% of the well's q95-q5 range)
        (dry well censored at sensor depth: Dhule CGWNAG2183 at -54.6 / -13.1 / -5.2 m)
    """
    roll = s.rolling(PLATEAU_WIN, center=True, min_periods=20)
    iqr = roll.quantile(0.75) - roll.quantile(0.25)
    typical = iqr.median()
    if not np.isfinite(typical) or typical <= PLATEAU_NORMAL_IQR:
        return pd.Series(False, index=s.index)
    flat = (iqr < PLATEAU_IQR).fillna(False)
    flat = flat.astype(int).rolling(PLATEAU_WIN, center=True, min_periods=1).max().astype(bool)
    full_rng = s.quantile(0.95) - s.quantile(0.05)
    r60 = s.rolling(2 * PLATEAU_WIN, center=True, min_periods=40)
    # floors only: a flat 60-day window in the DEEPER half of the well's range (a dry well
    # reading its sensor depth). A flat window near the shallow end is a full aquifer at
    # its spill level — real, kept.
    deep = r60.median() < s.median()
    narrow = (((r60.max() - r60.min()) < min(0.5, 0.05 * full_rng)) & deep).fillna(False)
    narrow = narrow.astype(int).rolling(2 * PLATEAU_WIN, center=True, min_periods=1).max().astype(bool)
    return (flat | narrow) & s.notna()


def detect_steps(s: pd.Series, others: pd.DataFrame, sigma_d: float):
    """Walk a station's daily series against the LAST ACCEPTED value.

    Returns (accepted series, segment id series, counts). `others` = the district's
    other stations (calendar-indexed, already Hampel/plateau cleaned) for the
    co-movement test.
    """
    vals = s.dropna()
    idx = list(vals.index)
    out = pd.Series(np.nan, index=s.index)
    seg = pd.Series(np.nan, index=s.index)
    if not idx:
        return out, seg, {"rejected": 0, "breaks": 0, "comove": 0}
    oth = others.ffill(limit=3) if others is not None and others.shape[1] else None
    k, last_d, last_v = 0, idx[0], vals.iloc[0]
    out[last_d], seg[last_d] = last_v, k
    n_rej = n_brk = n_com = 0
    for j in range(1, len(idx)):
        t, x = idx[j], vals.iloc[j]
        gap = (t - last_d).days
        thr = max(STEP_MIN, STEP_SIGMA_K * sigma_d * np.sqrt(gap))
        jump = x - last_v
        if abs(jump) <= thr:
            out[t], seg[t], last_d, last_v = x, k, t, x
            continue
        # (1) did the district's other wells move the same way over the same interval?
        comove, has_others = False, False
        if oth is not None:
            ch = (oth.loc[t] - oth.loc[last_d]).dropna()
            if len(ch):
                has_others = True
                c = float(np.median(ch))
                comove = np.sign(c) == np.sign(jump) and abs(c) >= COMOVE_FRAC * abs(jump)
        if comove:
            n_com += 1
            out[t], seg[t], last_d, last_v = x, k, t, x
            continue
        # (2) does the new level persist?
        fut = vals.iloc[j + 1:j + 1 + LOOKAHEAD]
        persists = len(fut) >= 3 and (np.abs(fut - x) < np.abs(fut - last_v)).mean() >= PERSIST_FRAC
        if persists:
            # Only a jump that appears across a reporting OUTAGE is treated as a
            # recalibration (new segment, own offset) — e.g. Akola CGWBNG0501 came back
            # from 24 days of placeholders 9.2 m off. A persistent jump without an
            # outage is kept as real movement: splitting there removes real signal
            # (an earlier draft split one noisy Jalgaon well 34 times).
            if gap >= OUTAGE_DAYS:
                k += 1
                n_brk += 1
            out[t], seg[t], last_d, last_v = x, k, t, x
            continue
        n_rej += 1                         # transient fault: reject, keep the last accepted value
    return out, seg, {"rejected": n_rej, "breaks": n_brk, "comove": n_com}


def drift_r2(v: np.ndarray, t: np.ndarray):
    ss = ((v - v.mean()) ** 2).sum()
    if ss <= 0:
        return 0.0, 0.0
    coef, *_ = np.linalg.lstsq(np.vstack([t, np.ones_like(t)]).T, v, rcond=None)
    r2 = 1.0 - (((v - (coef[0] * t + coef[1])) ** 2).sum() / ss)
    return float(r2), float(v.max() - v.min())


BETA_PRIOR_DAYS = 365      # shrinkage of a segment's amplitude toward 1 (pseudo-days of evidence)
# Chosen by audit_groundwater_target.py (model-free, 2026-10-02):
#                 jumps>2m   LOSO r30
#   v3 target        329        -
#   median_anom      267      0.445
#   offset           199      0.442
#   factor           164      0.430
#   factor_med       193      0.458   <- fewest jumps among methods that do not lose LOSO
AGG_METHOD = "factor_med"


def aggregate(X: pd.DataFrame, method: str = AGG_METHOD):
    """Combine pseudo-stations into one district signal D(t) (metres, mean ~0).

    X: calendar-indexed, one column per pseudo-station (segment), NaN when not reporting.
    Returns (D, mu, beta) with x_p(t) ~ mu_p + beta_p * D(t).

      median_anom : v3 — each station's anomaly from its OWN mean, median across stations.
                    A station whose record covers only part of the seasonal cycle gets a
                    biased mean, so it shifts the district level when it joins or leaves.
      offset      : two-way fixed effects, beta = 1. Offsets mu_p are estimated jointly with
                    the common signal, so partial-season segments are referenced correctly.
      factor      : as offset but each segment also gets an amplitude beta_p (regression on
                    the common signal), shrunk toward 1 by BETA_PRIOR_DAYS and clipped.
    """
    cols = list(X.columns)
    V = X.values.astype(np.float64)
    avail = np.isfinite(V)
    if method == "median_anom":
        mu = np.nanmean(V, axis=0)
        D = np.nanmedian(V - mu, axis=1)
        return pd.Series(D, index=X.index), pd.Series(mu, index=cols), pd.Series(1.0, index=cols)

    mu = np.nanmedian(V, axis=0)
    beta = np.ones(len(cols))
    D = np.full(len(X), np.nan)
    scaled = method in ("factor", "factor_med")
    for _ in range(FACTOR_ITERS):
        Z = (V - mu) / beta
        if method == "factor_med":
            # each segment's de-offset, de-scaled reading is one vote; the median of votes
            # is robust to a single misbehaving well and gives every well equal say
            D = np.nanmedian(Z, axis=1)
        else:
            w = np.where(avail, beta ** 2, 0.0)
            num = np.where(avail, w * Z, 0.0).sum(1)
            den = w.sum(1)
            D = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        D = D - np.nanmean(D)
        for p in range(len(cols)):
            m = avail[:, p] & np.isfinite(D)
            n = int(m.sum())
            if n == 0:
                continue
            if scaled and n >= 60 and np.var(D[m]) > 1e-9:
                b_hat = max(np.polyfit(D[m], V[m, p], 1)[0], 1e-3)
                b = float(np.exp(n / (n + BETA_PRIOR_DAYS) * np.log(b_hat)))
                beta[p] = float(np.clip(b, *BETA_CLIP))
            mu[p] = float(np.mean(V[m, p] - beta[p] * D[m]))
        if scaled:
            s = np.median(beta)
            beta, D = beta / s, D * s
    return pd.Series(D, index=X.index), pd.Series(mu, index=cols), pd.Series(beta, index=cols)


# ─── raw readings -> cleaned station-day means ────────────────────────────────
def load_raw():
    df = pd.read_csv(INPUT, low_memory=False, parse_dates=["dataTime"])
    print(f"  Raw rows: {len(df):,} | stations {df['stationCode'].nunique()} | districts {df['district'].nunique()}")
    return df


def clean_readings(df, log):
    n0 = len(df)
    df = df[(df["dataAcquisitionMode"] == "Telemetric") & (df["stationStatus"] == "Active")].copy()
    df = df.dropna(subset=["dataValue", "district"])
    df = df.drop_duplicates(subset=["stationCode", "dataTime"], keep="last")
    print(f"  Telemetric panel: {df['stationCode'].nunique()} stations, {len(df):,} readings "
          f"(manual-survey wells dropped: {n0 - len(df):,} rows)")
    for name, m in [("exact zero", df["dataValue"] == 0),
                    (f"|v| > {MAX_ABS_DEPTH:.0f} m", df["dataValue"].abs() > MAX_ABS_DEPTH),
                    ("placeholder +1.000", df["dataValue"].isin(PLACEHOLDERS))]:
        for code, n in df.loc[m].groupby("stationCode").size().items():
            log.append({"station": code, "step": name, "removed": int(n), "unit": "readings"})
        print(f"  dropped {name:<20}: {int(m.sum()):>6,} readings")
        df = df[~m]

    # sign: drop minority-sign readings (fault periods) and their near neighbours
    df = df.sort_values(["stationCode", "dataTime"]).reset_index(drop=True)
    neg_frac = df.groupby("stationCode")["dataValue"].apply(lambda v: (v < 0).mean())
    maj_neg = df["stationCode"].map(neg_frac >= 0.5)
    minority = np.where(maj_neg, df["dataValue"] > 0, df["dataValue"] < 0)
    df["_minority"] = minority
    guard = np.zeros(len(df), bool)
    for code, g in df[df["_minority"]].groupby("stationCode"):
        rows = df.index[df["stationCode"] == code]
        t_all = df.loc[rows, "dataTime"].values
        t_bad = g["dataTime"].values
        pos = np.searchsorted(t_bad, t_all)
        dl = np.abs(t_all - t_bad[np.clip(pos - 1, 0, len(t_bad) - 1)])
        dr = np.abs(t_bad[np.clip(pos, 0, len(t_bad) - 1)] - t_all)
        near = np.minimum(dl, dr) <= np.timedelta64(SIGN_GUARD_DAYS, "D")
        guard[rows] = near
        log.append({"station": code, "step": "minority-sign fault + guard",
                    "removed": int(near.sum()), "unit": "readings"})
    print(f"  dropped minority-sign : {int(minority.sum()):>6,} readings "
          f"(+{int(guard.sum() - minority.sum()):,} within +-{SIGN_GUARD_DAYS} d) across "
          f"{df.loc[df['_minority'], 'stationCode'].nunique()} stations — no station dropped for sign")
    df = df[~guard].drop(columns="_minority")
    df["dataValue"] = -df["dataValue"].abs()

    # stuck logger: long runs of identical raw readings
    df = df.sort_values(["stationCode", "dataTime"]).reset_index(drop=True)
    run_id = (df["dataValue"] != df.groupby("stationCode")["dataValue"].shift()).cumsum()
    run_len = df.groupby(run_id)["dataValue"].transform("size")
    stuck = run_len >= STUCK_RUN
    for code, n in df.loc[stuck].groupby("stationCode").size().items():
        log.append({"station": code, "step": f"stuck run >= {STUCK_RUN}", "removed": int(n), "unit": "readings"})
    print(f"  dropped stuck runs    : {int(stuck.sum()):>6,} readings in "
          f"{df.loc[stuck, 'stationCode'].nunique()} stations")
    df = df[~stuck]
    df["date"] = df["dataTime"].dt.normalize()
    return df


def clean_station_days(df, log):
    """Hampel + plateau per station on calendar-indexed daily means; drop dead / thin stations."""
    sd = df.groupby(["stationCode", "district", "date"])["dataValue"].mean().reset_index()
    series, sig = {}, {}
    dist_of = sd.drop_duplicates("stationCode").set_index("stationCode")["district"].to_dict()
    for code, g in sd.groupby("stationCode"):
        s = g.set_index("date")["dataValue"].sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        sig_d = robust_daily_sd(s)
        spk = hampel_two_sided(s, sig_d)
        s = s.mask(spk)
        pla = plateau_mask(s)
        s = s.mask(pla)
        if spk.sum():
            log.append({"station": code, "step": "hampel spike", "removed": int(spk.sum()), "unit": "days"})
        if pla.sum():
            log.append({"station": code, "step": "plateau / censored", "removed": int(pla.sum()), "unit": "days"})
        v = s.dropna()
        if len(v) and (v.quantile(0.75) - v.quantile(0.25)) < DEAD_IQR:
            log.append({"station": code, "step": "dead sensor (IQR)", "removed": len(v), "unit": "days"})
            continue
        if len(v) < MIN_GOOD_DAYS:
            log.append({"station": code, "step": f"< {MIN_GOOD_DAYS} good days", "removed": len(v), "unit": "days"})
            continue
        series[code], sig[code] = s, sig_d
    print(f"  station-days: Hampel + plateau done; {len(series)} stations kept")
    return series, sig, dist_of


def segment_stations(series, sig, dist_of, log):
    """Step detection with co-movement test -> pseudo-station segments; drift filter per segment."""
    segs = {}           # (code, k) -> series
    by_dist = {}
    for code, d in dist_of.items():
        if code in series:
            by_dist.setdefault(d, []).append(code)
    for dist, codes in by_dist.items():
        cal = pd.date_range(min(series[c].index.min() for c in codes),
                            max(series[c].index.max() for c in codes), freq="D")
        panel = pd.DataFrame({c: series[c].reindex(cal) for c in codes})
        for c in codes:
            acc, sg, cnt = detect_steps(panel[c], panel.drop(columns=c), sig[c])
            for kk, v in [("step rejected", cnt["rejected"]), ("regime break", cnt["breaks"]),
                          ("big jump kept (co-moved)", cnt["comove"])]:
                if v:
                    log.append({"station": c, "step": kk, "removed": int(v) if kk == "step rejected" else 0,
                                "unit": "days" if kk == "step rejected" else f"{v} events"})
            for k in sorted(sg.dropna().unique()):
                v = acc[sg == k].dropna()
                if len(v) < MIN_SEGMENT_DAYS:
                    log.append({"station": c, "step": f"segment {int(k)} < {MIN_SEGMENT_DAYS} d",
                                "removed": len(v), "unit": "days"})
                    continue
                t = (v.index - v.index.min()).days.values.astype(float)
                r2, rng = drift_r2(v.values, t)
                if len(v) >= 50 and r2 >= DRIFT_R2 and rng > DRIFT_RANGE:
                    log.append({"station": c, "step": f"drift (R2 {r2:.2f}, range {rng:.0f} m)",
                                "removed": len(v), "unit": "days"})
                    continue
                segs[(c, int(k))] = v
    # a station must still have MIN_GOOD_DAYS across its surviving segments
    tot = pd.Series({key: len(v) for key, v in segs.items()}).groupby(level=0).sum()
    thin = set(tot[tot < MIN_GOOD_DAYS].index)
    for c in thin:
        log.append({"station": c, "step": f"< {MIN_GOOD_DAYS} days after segmentation",
                    "removed": int(tot[c]), "unit": "days"})
    segs = {k: v for k, v in segs.items() if k[0] not in thin}
    print(f"  segments: {len(segs)} pseudo-stations from {len({k[0] for k in segs})} stations")
    return segs


# ─── district aggregation ─────────────────────────────────────────────────────
def build_district(segs, dist_of, cal, method=AGG_METHOD, verbose=True):
    rows, station_rows = [], []
    by_dist = {}
    for key in segs:
        by_dist.setdefault(dist_of[key[0]], []).append(key)
    for dist, keys in sorted(by_dist.items()):
        X = pd.DataFrame({f"{c}#{k}": segs[(c, k)].reindex(cal) for c, k in keys})
        D, mu, beta = aggregate(X, method)
        # district reference level: one value per WELL (its segments' offsets weighted by
        # length), median over wells — a well split into several segments must not get
        # several votes (Dhule's shallow well would otherwise pull the level up ~30 m)
        seg_len = X.notna().sum()
        st = pd.DataFrame({"st": [c.split("#")[0] for c in X.columns], "mu": mu.values, "n": seg_len.values})
        st_mu = st.groupby("st").apply(lambda g: np.average(g["mu"], weights=np.maximum(g["n"], 1)))
        ref = float(np.median(st_mu.values))
        Z = (X - mu) / beta                                  # each segment's estimate of D (metres)
        n_real = X.notna().sum(1)
        stations_real = pd.DataFrame({c: X[[f"{c}#{k}" for cc, k in keys if cc == c]].notna().any(axis=1)
                                      for c in {c for c, _ in keys}}).sum(1)
        span = pd.DataFrame({col: (cal >= X[col].first_valid_index()) & (cal <= X[col].last_valid_index())
                             for col in X.columns}, index=cal)
        n_active = span.sum(1)
        need = np.maximum(1, np.ceil(OBS_FRAC * n_active))
        obs = (n_real >= need) & (n_real > 0) & D.notna()

        level = ref + D
        # district-level jump guard against the last ACCEPTED value (flag + blank, no neighbours)
        sd_d = robust_daily_sd(level.where(obs))
        last_t, last_v, n_flag = None, None, 0
        lv = level.values
        ob = obs.values.copy()
        for i in np.flatnonzero(ob):
            if last_t is None:
                last_t, last_v = i, lv[i]
                continue
            thr = max(DISTRICT_JUMP_MIN, DISTRICT_JUMP_K * sd_d * np.sqrt(i - last_t))
            if abs(lv[i] - last_v) > thr:
                ahead = lv[np.flatnonzero(ob)[(np.flatnonzero(ob) > i)][:5]]
                if len(ahead) and (np.abs(ahead - lv[i]) < np.abs(ahead - last_v)).mean() >= 0.6:
                    last_t, last_v = i, lv[i]                 # the new level persists: accept
                else:
                    ob[i] = False
                    n_flag += 1
                continue
            last_t, last_v = i, lv[i]
        jump_flag = obs.values & ~ob
        level = level.mask(jump_flag)
        obs = pd.Series(ob, index=cal)

        out = pd.DataFrame({
            "district": dist, "date": cal,
            "gwl_mean": level.values,
            "gwl_min": (ref + Z.min(axis=1)).mask(jump_flag).values,
            "gwl_max": (ref + Z.max(axis=1)).mask(jump_flag).values,
            "station_count": stations_real.reindex(cal).fillna(0).values,
            "n_active": n_active.values,
            "gwl_frac_real": (n_real / n_active.replace(0, np.nan)).fillna(0).round(3).values,
            "gwl_observed": obs.astype(int).values,
        })
        rows.append(out)
        for col in X.columns:
            v = X[col].dropna()
            station_rows.append(pd.DataFrame({"stationCode": col.split("#")[0], "segment": int(col.split("#")[1]),
                                              "district": dist, "date": v.index, "value": v.values,
                                              "mu": mu[col], "beta": round(float(beta[col]), 3)}))
        if verbose:
            print(f"  {dist:<12} segments {len(keys):>2} | beta {np.round(beta.values, 2).tolist()} | "
                  f"observed {obs.mean():.2f} | district-jump flags {n_flag}")
    return pd.concat(rows, ignore_index=True), pd.concat(station_rows, ignore_index=True)


def district_metadata(raw, keep_codes):
    r = raw[raw["stationCode"].isin(keep_codes)]
    meta = r.groupby("district").agg(
        aquifer_type=("wellAquiferType", lambda s: s.mode().iloc[0] if s.notna().any() else "Unknown"),
        well_depth_m=("wellDepth", "mean"),
        n_stations=("stationCode", "nunique"),
    ).reset_index()
    meta["well_depth_m"] = meta["well_depth_m"].round(1)
    return meta


def build_segments(verbose=True):
    """Stages 1-6. Returns (raw, segs, dist_of, log)."""
    log = []
    if verbose:
        print("Loading groundwater data ...")
    raw = load_raw()
    df = clean_readings(raw, log)
    series, sig, dist_of = clean_station_days(df, log)
    segs = segment_stations(series, sig, dist_of, log)
    return raw, segs, dist_of, log


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    CLEAN.mkdir(exist_ok=True)
    raw, segs, dist_of, log = build_segments()

    cal = pd.date_range(START, END, freq="D")
    print(f"\nDistrict aggregation ({AGG_METHOD}):")
    gw, station_daily = build_district(segs, dist_of, cal)

    keep_codes = {k[0] for k in segs}
    meta = district_metadata(raw, keep_codes)
    gw = gw.merge(meta, on="district", how="left")

    print("\nPer district:")
    summ = gw.groupby("district").agg(n_stations=("n_stations", "first"), observed=("gwl_observed", "mean"),
                                      mean_level=("gwl_mean", "mean"), std=("gwl_mean", "std"))
    print(summ.round(2).to_string())
    print(f"\nOverall observed: {gw['gwl_observed'].mean():.1%} | NaN gwl_mean: {gw['gwl_mean'].isna().mean():.1%}")

    out_cols = ["district", "date", "gwl_mean", "gwl_min", "gwl_max", "station_count", "gwl_observed",
                "n_active", "gwl_frac_real", "aquifer_type", "well_depth_m", "n_stations"]
    gw[out_cols].to_csv(OUTPUT, index=False)
    station_daily.to_csv(CLEAN / "groundwater_station_daily.csv", index=False)
    qc = pd.DataFrame(log)
    qc["district"] = qc["station"].map(raw.drop_duplicates("stationCode").set_index("stationCode")["district"])
    qc.to_csv(CLEAN / "station_qc_log.csv", index=False)
    print(f"\nSaved -> {OUTPUT}\n         {CLEAN / 'groundwater_station_daily.csv'}\n         {CLEAN / 'station_qc_log.csv'}")
    print("\nQC summary (events by step):")
    print(qc.groupby(["step", "unit"])["removed"].agg(["count", "sum"]).to_string())


if __name__ == "__main__":
    main()
