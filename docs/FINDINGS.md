# Findings

Empirical results that should outlive any one session. Each entry says what was
measured, on what data, and what decision it drove. Commits referenced are on
`main`; bead IDs refer to the local beads DB (`bd show <id>`).

Last updated: 2026-08-03.

---

## 1. The bot has no demonstrated edge, and the market is the reason

The single most important fact about this project. Measured three independent
ways, all converging:

| model | evaluation | Brier vs market |
|---|---|---:|
| pitcher_ks (Poisson GLM + NB tail) | 4,371 walk-forward rows | **+0.00675 worse** |
| player_hits (binomial, calibrated) | 7,157 rows, real asks + real outcomes | **+0.00316 worse** |
| market ask vs realized rate | both sets | market ≈ perfectly calibrated |

The Kalshi ask sits at or slightly **above** the realized rate (that's the
spread), so buying YES at ask requires *beating* the market, not matching it.
After fixing every estimator bug we found, both models moved *toward* the
market, not past it. `player_hits` now agrees with the market to within 1–2¢
(Ohtani 1+ hits: model 67% vs ask 67¢).

**Decision:** stop refining models to chase this gap (`ix6`, closed as a
negative result). Any future effort should target an *informational* advantage:
consensus prices from sharper venues (`slugger/consensus.py` scaffolding exists,
unused), or thin/illiquid markets (testable once a week of spread-instrumented
signals accumulates — instrumentation live since 2026-08-02).

---

## 2. Fees were 29% of all losses; price level dominates model quality

From 1,042 settled trades: **$40.24 of fees on $598.84 stake (6.7%), against a
total P&L of −$140.82.** The fee `ceil(0.07 × C × P × (1−P))` is fixed-ish in
notional but brutal as a share of stake at low prices:

| entry price | n | fee/stake |
|---|---:|---:|
| 0–20¢ | 819 | **7.7%** |
| 20–40¢ | 179 | 5.7% |
| 40–60¢ | 21 | 3.2% |
| 60–80¢ | 23 | **0.6%** |

819 of 1,042 trades were in the worst bucket. The real bar was never "beat the
market" — it was "beat the market by more than ~7% of stake plus half the
spread" at the prices we traded.

**Decisions:** net edge now computed as `gross − exact_fee(price) −
half_spread − 2¢ residual` (commit `0165637`); prefer higher-priced contracts
structurally; fee rate is per-series configurable on Kalshi
(`GET /series/{ticker}/fee_changes`) so check it before entering any new series.

---

## 3. Market making on Kalshi is closed off (bead `33s`)

Four independent blockers, each sufficient:

1. **Maker fills pay trading fees** — the fee-rounding docs state the fee
   accumulator applies "regardless of whether the fills are taker or maker."
2. **No liquidity incentives on our series** — 3,830 active programs,
   $263,588 in rewards (queried live via `GET /incentive_programs`), none on
   KXMLBKS/HIT/HR/HRR. Subsidies go to macro/tech markets.
3. **Capital** — incentive obligations are 1,000 contracts (~$200+/side
   notional) vs a $13.93 account.
4. **The bot cannot rest an order** — `limit_price_cents` returns
   `min(fair−buffer, ask)`; on any trade with edge the ask binds, so 774/774
   joinable fills were taker-priced.

---

## 4. Every strategy hid a mis-specified estimator behind a hand deflator

The recurring anti-pattern of the whole codebase. Three instances, three
different root causes:

| strategy | patch | what it was hiding | fix |
|---|---|---|---|
| pitcher_ks | `KS_LAMBDA_DEFLATOR=0.85` | OLS on log(count) recovers the **geometric** mean → λ biased −0.57 Ks/start; plus Poisson tail too thin for overdispersed counts (conditional var/mean = 1.129) | Poisson GLM via IRLS + `negbinom_ge` with fitted dispersion (`543cad9`, `a3b5d4d`) |
| player_hits | `HITS_LAMBDA_DEFLATOR=0.70` | Hits are **Binomial** (bounded AB), not Poisson — Poisson understated 1+ by 4pts and overstated 3+/4+ by 1.3–1.7× (the phantom longshot edge) | `binomial_ge`; multipliers scale per-AB probability (`0acb2f9`) |
| pitcher_ks (Ks bias, downstream) | Phase-0 gates "too tight" | A model biased 13% low can never report YES edge, so the 20¢ floor was unreachable **by construction**; edge≥X ROI got *worse* as X rose | fix the estimator, leave the gates |

**But not every deflator is a lie:** `HITS_LAMBDA_DEFLATOR=0.70` was re-tested
after the binomial fix against 7,170 real outcomes and is **still optimal**
(Brier 0.174 at 0.70 vs 0.193 at 1.0; undeflated model says 77% for a 62%
event). The blended-avg × WHIP × hard-hit × park stack overpredicts per-AB
probability by ~30% and the deflator corrects exactly that. Evidence is on the
constant in `models.py`. Rule: **re-derive, don't just delete.**

---

## 5. Calibration fit on traded-only outcomes is actively harmful

`player_hits` calibration was fit on Kalshi settlements — which only exist for
markets the bot *traded*, and it traded the ones showing the largest apparent
edge, i.e. the ones it most overestimated. Classic selection bias:

```
walk-forward Brier, 5,068 rows
  raw model, no calibration     0.17746
  traded-only curve             0.18679   <- worse than nothing
  unbiased curve (game logs)    0.17711
```

**Decision:** `backfill_outcomes` reconstructs outcomes from MLB game logs for
every *listed* market (pitcher_ks and player_hits). The curve went from 181
samples / 4 breakpoints (clamping everything above raw 37.5% to 31%) to 7,053
samples / 13 breakpoints spanning 2–78%. Related: `_interpolate` extrapolates
toward (0,0) and (100,100) outside the fitted domain instead of clamping —
clamping is only sound when the fitted domain covers the traded range.

---

## 6. Evaluation-window choice can mislead by an order of magnitude

The same pitcher_ks model scored **+0.00068** Brier deficit on the official
late-20% holdout and **+0.00675** on 4,371 walk-forward rows across the season
— a 10× difference from nothing but window choice. We quoted the flattering
number for half a day before catching it.

**Decisions:**
- `beats_market` (the strategy re-enable gate) requires the **whole bootstrap
  CI** to favour the model, and fails closed below 30 rows (`e9a0f49`).
- Point-estimate subset wins don't count either: the 9+ threshold showed
  +14.3% ROI (n=518) but bootstraps to CI [−16.2%, +46.6%].
- Estimate parameters, never tune them on the evaluation metric: hand-sweeping
  NB dispersion beat the fitted value on holdout Brier, and we shipped the
  fitted value anyway.

---

## 7. Negative results worth not re-running

- **pitcher_ks features (`ix6`)**: log(batters-faced), rest days, K-per-BF
  rate, pitch count, and BF-as-offset all fail to improve out-of-sample Brier;
  the exposure offset (predicted to be the biggest win) was second-worst,
  because prior-start BF is a weak proxy and recent_k already encodes exposure.
  The deficit is *widest* exactly where the strategy would trade (5+–8+
  thresholds, 30–60¢). Untried: Statcast CSW/whiff, lineup-level opponent K%.
- **Gate re-tuning for pitcher_ks (`ibu`)**: no gate configuration is
  profitable; the "edge" signal was measuring our own estimator bias.
- **Kalshi MLB title formats (`8r5`)**: all observed titles use the `N+` form;
  the richer "over 6.5"/"at least 9" parsers were unreachable and are deleted.
  `record_unparsed_title` in `signal_pipeline.py` is the tripwire if this ever
  changes.

---

## 8. Operational traps (each burned us once)

- **Sticky health latch (`ja6`)**: `StrategyHealthMonitor.disabled` is add-only
  by design (no intra-session flapping), so seeding from history must judge
  once from the final window — replaying with per-observation evaluation
  latches on the worst window ever and disabled *everything* at boot.
- **Silent fallbacks look like success**: no `logs/ks_model.json` → hand
  heuristic silently prices Ks; missing `fee_cost` on a settlement → $0
  recorded (27/1042); a broken `calibrate --fit` printed one line and carried
  on. All three now fail loudly. When adding a fallback, log the fact that it
  engaged.
- **`opp_k_rate=0.0` means "unknown"**, and only the hand model read it that
  way. The trained model has a real coefficient (largest in the model, 1.375),
  so a failed team fetch looked like a lineup that never strikes out (−30% λ).
  Serving substitutes the league average, matching training.
- **Point-in-time discipline**: season totals from the MLB API are current-day
  snapshots — features must be rebuilt from game logs sliced strictly before
  the start date (`team_k_rate_as_of`), and walk-forward splits must cut on
  date boundaries or same-day starts leak across.
- **The test suite silently depended on no model artifact existing** —
  `get_trained_ks_model` bound its path as a default argument (frozen at
  import). Now resolved at call time + autouse conftest fixture isolates tests
  from `logs/`.
- **Zero trades is ambiguous**: gate-rejection counters
  (`rejected_by_gate`) distinguish "model finds no edge" from "gates
  unreachable" from "model broken". Keep them.

---

## 9. Codebase conventions that encode decisions

- `STRATEGY_PIPELINE` is the only live registry; `ENABLED_STRATEGIES` can only
  narrow it (test-enforced). Retired strategies are **deleted**, with journal
  evidence preserved in `RETIRED_STRATEGIES` — git history is the archive.
- Dead code is kept only when re-enabling is cheaper than rewriting
  (`no_side`/`_evaluate_no_side` qualifies; nothing else did).
- Re-enable bar for any strategy: `beats_market=True` (bootstrap-significant)
  on walk-forward holdout, *then* re-derive gates from the model's own
  probabilities. Never widen gates to force trades — the below-band cells are
  longshots, where journal ROI was worst.

---

## 10. Kalshi prop microstructure is inversely aligned with model reliability

Audited 547 open MLB prop markets (2026-08-03). Median spread by cell, with the
gross edge required to clear the 10¢ net floor after half-spread and fee:

| cell | price | spread | gross needed | as % of price |
|---|---:|---:|---:|---:|
| hits Over 0.5 | 62¢ | 4¢ | 14¢ | 23% |
| hits Over 1.5 | 25¢ | 9¢ | 17¢ | 68% |
| hits Over 2.5 | 6¢ | 2¢ | 12¢ | **200%** |
| hits Over 3.5 | 3¢ | 1¢ | 12¢ | **400%** |
| Ks Over 4.5 | 45¢ | 8¢ | 16¢ | 36% |
| Ks Over 5.5 | 30¢ | 7¢ | 16¢ | 53% |
| Ks Over 6.5 | 20¢ | 6¢ | 15¢ | 75% |

Two structural facts:

1. **The tight-spread cells are the ones we shouldn't trade.** Longshot hit
   props (Over 2.5/3.5) quote 1–2¢ wide, but carry the worst fee drag
   (§2) and were where the model was least reliable (§4). The cells where the
   model is most trustworthy carry 4–8¢ spreads.
2. **The 10¢ edge floor is absolute, so it is a de facto longshot ban** — 400%
   relative edge at 3¢, 23% at 62¢. Defensible given §2, but it should be
   deliberate rather than an artifact of measuring in cents. Bead filed.

Useful schema details found while auditing (`GET /markets`):
- **`floor_strike` is already in book form** (0.5, 1.5, 2.5 …), so the
  Kalshi-`N+` ↔ book-`Over N−0.5` translation needs no code — read the field
  instead of parsing the ticker suffix.
- **`custom_strike.baseball_player` is a UUID** (plus `baseball_team`), very
  likely Sportradar given Kalshi's live-data API uses Sportradar. If an odds
  provider keys on the same IDs, player mapping is an exact join instead of the
  name matching we currently limp along with. Verify before writing a matcher.
- `liquidity_dollars` reads `0.0000` on every market; use `open_interest_fp`
  as the depth proxy.
- p90 spreads are far worse than medians (23–25¢ hits, 48–76¢ Ks), so any
  spread gate must be per-market, not per-cell.
