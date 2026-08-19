# Findings

Empirical results that should outlive any one session. Each entry says what was
measured, on what data, and what decision it drove. Commits referenced are on
`main`; bead IDs refer to the local beads DB (`bd show <id>`).

Last updated: 2026-08-19.

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

---

## 11. Consensus/de-vig is dead: Kalshi's spread is ~3× the book's vig

Gate 0 run 2026-08-03 with a real Odds API key, one game (TOR@HOU), 18
de-viggable cells joined to live Kalshi quotes. Cost: 4 of 500 credits.

```
book vig (DK/BetMGM/FD/BetRivers)   median  7.6%   (range 7.0-8.1)
Kalshi spread, same cells           median   22c   (range 7-91)

de-vigged book fair vs Kalshi MID   median  +7.9c   <- ~half the spread
de-vigged book fair vs Kalshi ASK   median  -2.9c   best +1.8c
required edge (halfspread+fee+10c)          16-27c
cells clearing it                            0 / 18
```

**The de-vigged book price lands almost exactly on Kalshi's mid.** The two
venues agree on fair value; there is no disagreement to arbitrage. Kalshi's ask
sits above fair by roughly half the spread — which is what an ask is supposed to
do.

### Correction to earlier reasoning in this file

§1 argued consensus was the most promising direction because "books charge 6–10%
vig, Kalshi charges ~2% fee plus spread, so the venues have different cost
structures and should persistently disagree on displayed price." **That was
wrong.** It treated Kalshi's cost as fee-dominated. On these props Kalshi's
*spread* is 10–45%, which dwarfs both its fee and the book's vig. The
cheap-fee venue is the expensive venue once you cross its spread. Fee drag (§2)
is real but it is the second-order cost here; the spread is first-order.

### Supporting details

- **`_alternate` ladders are Over-only** — `sides=['Over']` at every alternate
  threshold, so they cannot be de-vigged (no overround to remove). Only the main
  lines are two-sided: hits 0.5/1.5 and Ks 4.5. That eliminates most of the
  ladder, including the Ks 5.5/6.5 cells §10 identified as targets.
- **No shared player ID.** `description` is a plain name; `sid` is
  bookmaker-internal. The Sportradar-join hope from §10 is dead — name matching
  with unicode normalisation is required ("Yandy Díaz" → "yandy diaz").
- **"Consensus" is mostly one book.** 13 of 18 cells had a single two-sided
  quote (DraftKings). It is DK's opinion, not a consensus.
- **Garbage quotes masquerade as edge.** Varsho 1.5 showed bid 2¢ / ask 93¢ — a
  91¢ spread — presenting as −37.7¢ "edge". Any live use needs a hard
  per-market spread gate, and the apparent outliers will almost always be
  illiquidity rather than opportunity.
- The one place the arithmetic works is **earning mid rather than paying the
  ask** (+7.9¢). That is market making, closed off in §3 for independent
  reasons (fees on maker fills, no liquidity incentive on these series,
  1000-contract obligations vs a $13.93 account, and `limit_price_cents` cannot
  rest an order). The story is consistent: the only viable edge on this venue
  requires providing liquidity, and we cannot.

---

## 12. Parlay markets are grossly mispriced — and the profitable side cannot be taken

Ran the series calibration audit (2026-08-03): pull settled markets, compare
last traded price to realised outcome, per series. No model, no odds key, no
capital required. 12,000 settled markets collected; the feed is recency-ordered
so only 4 series appeared, but two had enough data to test — both multivariate
"combo"/parlay series.

`KXMVESPORTSMULTIGAMEEXTENDED` (n=3029) and `KXMVECROSSCATEGORY` (n=1093) show
enormous systematic **overpricing**:

| last price | n | priced | ACTUAL | bias |
|---|---:|---:|---:|---:|
| 0–10¢ | 2137 | 2.9% | **0.0%** | +2.9 |
| 10–20¢ | 416 | 14.0% | **1.0%** | **+13.0** |
| 20–30¢ | 162 | 24.5% | **5.6%** | **+18.9** |
| 30–40¢ | 115 | 34.3% | **7.8%** | **+26.5** |
| 40–50¢ | 57 | 45.4% | **17.5%** | **+27.8** |
| 50–60¢ | 37 | 54.2% | **29.7%** | **+24.5** |
| 70–100¢ | 79 | — | — | −3 to −11 (underpriced) |

That is the classic retail parlay bias, and it is 20–30 points wide — far larger
than anything in MLB props. The second series replicates it independently
(+12 to +31).

### Why it is not tradeable by us

The exploitable direction is short/NO. Checked 3000 live combo markets: **the
book is entirely one-sided.**

```
yes_ask + no_bid = 1.0000 exactly    (they are the same order)
no_ask  = 1.0000   -> cannot buy NO at any sensible price
yes_bid = 0.0000   -> cannot sell YES
```

2756 of 3000 markets had *only* a NO bid. The one executable action is **buying
YES — the side overpriced by 20–30 points.** The profitable side can only be
*posted*, never taken.

Whoever is posting those offers is running exactly this trade: selling
overpriced parlays to retail. That is also why §3's incentive query found
`KXMVESPORTSMULTIGAMEEXTENDED` among the subsidised series — Kalshi pays to
attract that liquidity.

### The conclusion this session keeps reaching

Three independent routes — MLB prop modelling (§1, §7), cross-venue consensus
(§11), and now a market-wide mispricing screen (§12) — all terminate at the same
wall: **the available edge on this venue belongs to liquidity providers, and
provision requires capital and resting-order logic we do not have** (§3:
1000-contract obligations, fees on maker fills, `limit_price_cents` cannot rest).

That is a coherent finding rather than a series of failures. It also sharpens
what would actually change the picture, in order of leverage:
1. Capital and a two-sided quoting engine (turns §12 from observation into a
   business).
2. A series that is **both** two-sided **and** mispriced. MLB props are
   two-sided but efficient; parlays are mispriced but one-sided. The audit
   method in this section is the cheap screen for finding one — it needs a
   proper series enumeration, since the settled feed is recency-dominated.
3. Nothing else measured this session moved the needle.

### Caveats on the method

- `last_price` is the last *trade*, not an executable quote. Good enough as a
  screen; not proof of tradeability. §12's own conclusion came from checking
  live quotes separately, which is the step that mattered.
- Requires `result in (yes,no)`, `volume > 0`, `0 < last_price < 100`.
- Favourite–longshot bias is near-universal in prediction markets. Finding it is
  not finding edge; the magnitude has to beat spread plus fees, and the side has
  to be executable.

---

## 13. Incentivised-series audit: every quotable series is well calibrated

The final screen (2026-08-03): if a series is *both* two-sided *and* mispriced,
market making there has a subsidised learning curve. Scanned all 199 series in
Kalshi's incentive program (the only clean enumeration available —
`/series/list` 404s and the settled feed is recency-dominated), 12,903 usable
settled markets, 24 series with n≥100.

Pre-committed kill criterion: report negative unless some series shows ≥10pt
overall bias or a ≥15pt bucket on n≥20. **Zero candidates.**

| series (top by n) | n | price | actual | bias | Brier |
|---|---:|---:|---:|---:|---:|
| KXTRUMPMENTION | 1000 | 41.4¢ | 41.2% | +0.2 | 0.0002 |
| KXTEMPAUSH (weather) | 917 | 62.8¢ | 62.9% | −0.1 | 0.0008 |
| KXAAAGASD (gas price) | 989 | 53.4¢ | 52.1% | +1.3 | 0.0207 |
| KXUSDJPY | 375 | 15.2¢ | 12.8% | +2.4 | 0.0185 |
| KXUST2AD (treasuries) | 186 | 59.3¢ | 55.4% | +4.0 | 0.0395 |
| …19 more, all within ±4pt | | | | | |

Worst single bucket anywhere: +11.2pt on n=47 (KXAAAGASD 10–20¢) — noise-sized.
Weather series are calibrated to a tenth of a point across 3,400+ markets;
mention markets have Briers of 0.0001–0.0002 (near-deterministic and priced as
such).

Confirms the anti-correlation §12 predicted: **the incentive program works.**
Paid quoters produce calibrated books. Mispricing survives only where no one is
paid to quote (parlays, §12) — and there the book is one-sided by construction.

### Session verdict

Every route is now measured, none is tradeable for us:

| route | finding | why not tradeable |
|---|---|---|
| MLB prop modelling | market efficient (§1) | 0.007 Brier behind, friction on top |
| Cross-venue consensus | venues agree (§11) | fair sits at Kalshi's ask |
| Parlay mispricing | 20–30pt (§12) | one-sided book; can only post |
| Incentivised series | all calibrated (§13) | professionals already quote them |

The remaining edge on this venue is liquidity provision, which requires capital
and a quoting engine (§3, §12). The bot should not trade
(`DRY_RUN=true` permanently); it remains a sound measurement instrument.

---

## 14. In-game overreaction: the market is done reacting before we know (bead `5vo`)

The Phase 1 thesis was that Kalshi overshoots on salient in-game events and
reverts, tradeably. Measured on the first full recorded slate
(**2026-08-18: 15 games, 18.5M Kalshi messages / 6.4 GB, 1,118 plays, 294
with |ΔWP| ≥ 3¢**), against the WP anchor from `d4p`. Reproduce with
`python3 scripts/overshoot_analysis.py 2026-08-18`.

### The market is fully repriced ~25 seconds before our feed delivers the play

Fraction of the market's total repricing already complete, timed off MLB's own
`about.endTime` for the play:

| t − endTime | −20s | −10s | −5s | **0s** | +5s | +30s |
|---|---:|---:|---:|---:|---:|---:|
| median complete | 0.00 | 0.40 | 0.74 | **0.91** | **1.00** | 1.00 |

Our GUMBO poller receives that play at a **median +25.3s** (p75 37s, p90 48s).
So by the time we learn a home run happened, the price has been at its new
level for half a minute.

This is not a clock artifact: our `recv_ts` vs Kalshi's exchange timestamps,
across 578k book updates, is **+0.03s median** (p99 +0.25s). It is also not a
polling artifact — the poller runs every 3s, and MLB's `endTime` is itself
~10s behind the physical play, which is why the market is already 40% moved at
`endTime − 10s`.

### What is left to capture, as a function of latency

Median |total market move| on these events is 4.0¢:

| our latency vs endTime | fraction left | cents left |
|---|---:|---:|
| −10s (impossible: ahead of MLB) | 0.78 | 3.14¢ |
| −5s (impossible) | 0.45 | 1.79¢ |
| 0s (perfect, unattainable) | 0.03 | **0.12¢** |
| +5s and beyond | 0.00 | **0.00¢** |
| +25s (**what we have**) | 0.00 | 0.00¢ |

Round-trip cost is 1–2¢ (fees per §13/`bvc`) plus any spread. Even a
*perfect zero-latency* feed leaves 0.12¢ against a 1¢ floor. Buying a faster
feed does not fix this; we would have to be ~5 seconds **ahead** of MLB's own
scoring timestamp, i.e. an in-park real-time feed.

### There is no overshoot to fade anyway

Signed excess move (market move − anchor move, in the event's direction;
positive = overshoot):

| t − endTime | 0s | +10s | +30s | +60s | +120s |
|---|---:|---:|---:|---:|---:|
| mean ¢ | −5.12 | −4.91 | −4.77 | −4.65 | −4.01 |
| share > 0 | 11% | 12% | 13% | 14% | 16% |

The sign is *negative and flat*: the market moves **less** than the anchor and
then stays there. Only 11–16% of events overshoot at all, and there is no decay
toward zero — no reversion to trade. Regression across all 506 clean events:

```
market move = −0.15 + 0.321 × anchor move    (se 0.034, t = 9.6)
```

The anchor moves ~3× as far as the market. The parsimonious reading is that the
WP model over-responds — it has no team, pitcher or lineup input, so it prices
every comeback as league-average — rather than that the market under-responds.
The slope is worst exactly where the model knows least (Single 0.11, Walk 0.23)
and best on outs, where game state is nearly sufficient (Strikeout 0.62,
Flyout 0.57).

### The trade itself, simulated

Enter at feed-receipt time (the earliest instant we could act), fade the
market/anchor residual, exit after a hold. **Gross of fees:**

| fill | \|residual\| | hold | n | mean | t | min detectable edge |
|---|---:|---:|---:|---:|---:|---:|
| mid (unattainable) | ≥3¢ | 60s | 150 | −0.08¢ | −0.5 | 0.30¢ |
| mid | ≥5¢ | 60s | 95 | −0.15¢ | −0.7 | 0.41¢ |
| exec (pay spread) | ≥3¢ | 60s | 150 | −0.38¢ | −1.9 | 0.41¢ |
| exec | ≥5¢ | 60s | 95 | −0.40¢ | −1.4 | 0.59¢ |

Zero at every threshold and horizon, negative once you pay the spread. The
sample is small but **not underpowered for the question asked**: it could have
resolved a 0.3–0.6¢ edge at t = 2, and the bar to clear is 1–2¢.

Liquidity was never the constraint — game-winner spreads are p50 **1¢**, p75 2¢.

**Decision:** kill criterion fired. `4g6` (live in-game mean-reversion trader)
closed unbuilt. `162` (isotonic recalibration of the WP model) closed too: it
addresses level calibration, but the gap here is *response magnitude*, and no
amount of WP accuracy buys back 25 seconds.

The reusable parts survive: `slugger/recorder/replay.py` (order-book
reconstruction from the raw feed) and `scripts/overshoot_analysis.py` re-run on
any recorded slate in ~1 minute. Any future in-game idea must first answer the
latency question, because on this venue it dominates every model question.
