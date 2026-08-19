"""Throwaway analysis: does Kalshi charge the taker fee on MAKER fills?

READ-ONLY. Uses get_fills / get_settlements only. Never places or cancels orders.

Method:
  1. Pull all fills (paginated via cursor) and all settlements.
  2. Report raw fill record fields.
  3. Classify each fill maker/taker: prefer explicit is_taker field; fall back
     to journal ask_at_entry + execution.classify_fill_role.
  4. Compare actual fee (per-fill if present, else per-ticker settlement
     fee_cost) vs taker formula ceil(0.07 * P * (1-P)) per contract.
"""
import json
import math
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slugger.config import Config
from slugger.models import kalshi_fee_cents_per_contract
from slugger.execution import classify_fill_role

JOURNAL = Path(__file__).resolve().parent.parent / "logs" / "journal.jsonl"


def to_cents(v):
    """Convert a Kalshi price-ish field to integer cents."""
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        f = float(v)
        return int(round(f * 100)) if 0 < f <= 1.0 else int(round(f))
    if isinstance(v, str):
        try:
            f = float(v)
        except ValueError:
            return 0
        if "." in v or f <= 1.0:
            return int(round(f * 100))
        return int(round(f))
    return 0


def to_dollars(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def paginate(client, path, key, limit=200, extra=None):
    out, cursor = [], None
    while True:
        params = {"limit": limit}
        if extra:
            params.update(extra)
        if cursor:
            params["cursor"] = cursor
        data = client._get(path, params=params)
        batch = data.get(key, []) or []
        out.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not batch:
            break
    return out


def main():
    cfg = Config.from_env()
    client = cfg.create_kalshi_client()

    fills = paginate(client, "/portfolio/fills", "fills")
    settlements = paginate(client, "/portfolio/settlements", "settlements")
    print(f"Pulled {len(fills)} fills, {len(settlements)} settlements\n")

    # ── (a) raw field inventory ──────────────────────────────────────────
    field_counts = Counter()
    for f in fills:
        field_counts.update(f.keys())
    print("=== FILL RECORD FIELDS (field: count present) ===")
    for k, c in sorted(field_counts.items()):
        print(f"  {k}: {c}")
    if fills:
        print("\nSample fill record:")
        print(json.dumps(fills[0], indent=2, default=str))
        print("\nSample fill record (last):")
        print(json.dumps(fills[-1], indent=2, default=str))

    if settlements:
        print("\n=== SETTLEMENT RECORD FIELDS ===")
        sc = Counter()
        for s in settlements:
            sc.update(s.keys())
        for k, c in sorted(sc.items()):
            print(f"  {k}: {c}")
        print("\nSample settlement record:")
        print(json.dumps(settlements[0], indent=2, default=str))

    # ── load journal for ask_at_entry fallback ───────────────────────────
    journal_by_ticker = {}
    if JOURNAL.exists():
        for line in JOURNAL.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "trade":
                # keep last trade per ticker (most have one)
                journal_by_ticker.setdefault(r["ticker"], []).append(r)

    # ── (b)(c) classify + fee comparison ─────────────────────────────────
    has_is_taker = any("is_taker" in f for f in fills)
    fee_fields = [k for k in field_counts if "fee" in k.lower()]
    print(f"\nExplicit is_taker present: {has_is_taker}")
    print(f"Per-fill fee-ish fields: {fee_fields}")

    rows = []  # (ticker, role, role_src, count, price_c, fee_actual_c or None, fee_formula_c)
    for f in fills:
        ticker = f.get("ticker", "")
        side = f.get("side", "")
        count = float(f.get("count_fp", f.get("count", 0)) or 0)
        # price of the contract actually bought (side-specific)
        yes_c = to_cents(f.get("yes_price", f.get("yes_price_dollars")))
        no_c = to_cents(f.get("no_price", f.get("no_price_dollars")))
        px = yes_c if side == "yes" else (no_c if side == "no" else yes_c or no_c)
        if px <= 0:
            px = to_cents(f.get("price"))

        # role
        if "is_taker" in f:
            role = "taker" if f["is_taker"] else "maker"
            src = "is_taker"
        else:
            jrs = journal_by_ticker.get(ticker, [])
            ask = jrs[-1].get("ask_cents", 0) if jrs else 0
            limit_c = jrs[-1].get("price_cents", 0) if jrs else 0
            role = classify_fill_role(limit_c, px, ask)
            src = "heuristic"

        # per-fill fee if the API exposes one
        fee_actual = None
        for k in fee_fields:
            d = to_dollars(f.get(k))
            if d is not None:
                fee_actual = d * 100.0  # cents
                break

        fee_formula = kalshi_fee_cents_per_contract(px) * count
        rows.append((ticker, role, src, count, px, fee_actual, fee_formula))

    # alternative formula candidates, checked per fill against actual fee:
    #   A: per-contract ceil-to-cent at 0.07 (models.py)
    #   B: total 0.07*P*(1-P)*C, ceil to $0.0001
    #   C: total 0.035*P*(1-P)*C, ceil to $0.0001
    def alt_fees(px, count):
        p = px / 100.0
        base = p * (1 - p) * count
        return {
            "A_percontract_ceil_7pct": kalshi_fee_cents_per_contract(px) * count,
            "B_total_7pct_ceil4dp": math.ceil(0.07 * base * 10000) / 100.0,
            "C_total_3.5pct_ceil4dp": math.ceil(0.035 * base * 10000) / 100.0,
            "D_total_7pct_ceilcent": math.ceil(0.07 * base * 100),
        }

    print("\n=== FORMULA-VARIANT MATCH RATES (per fill, actual fee in cents) ===")
    for role in ("maker", "taker"):
        sub = [(px, c, fa) for _, r, _, c, px, fa, _ in rows if r == role and fa is not None]
        if not sub:
            continue
        match = Counter()
        for px, c, fa in sub:
            for name, val in alt_fees(px, c).items():
                if abs(fa - val) < 0.005:
                    match[name] += 1
            if fa == 0:
                match["ZERO"] += 1
        print(f"  {role} (n={len(sub)}): " +
              ", ".join(f"{k}={v}" for k, v in sorted(match.items())))

    role_counts = Counter(r[1] for r in rows)
    src_counts = Counter(r[2] for r in rows)
    print(f"\n=== ROLE CLASSIFICATION ===")
    print(f"roles: {dict(role_counts)}   (source: {dict(src_counts)})")

    # per-fill fee comparison if available
    if any(r[5] is not None for r in rows):
        print("\n=== PER-FILL FEE vs FORMULA (fee field on fill records) ===")
        agg = defaultdict(lambda: [0, 0.0, 0.0, 0])  # fills, actual_c, formula_c, zero-fee fills
        for _, role, _, count, px, fee_a, fee_f in rows:
            if fee_a is None:
                continue
            a = agg[role]
            a[0] += 1
            a[1] += fee_a
            a[2] += fee_f
            if fee_a == 0:
                a[3] += 1
        for role, (n, act, form, zn) in sorted(agg.items()):
            print(f"  {role:8s} fills={n:5d}  actual_fee={act/100:8.2f}$  "
                  f"formula_fee={form/100:8.2f}$  zero-fee fills={zn}")
        # exact-match rate per role
        for role in sorted(set(r[1] for r in rows)):
            sub = [r for r in rows if r[1] == role and r[5] is not None]
            exact = sum(1 for r in sub if abs(r[5] - r[6]) < 0.5)
            zero = sum(1 for r in sub if r[5] == 0)
            if sub:
                print(f"  {role:8s} exact-formula-match: {exact}/{len(sub)}   zero-fee: {zero}/{len(sub)}")
    else:
        print("\nNo per-fill fee field. Falling back to settlement-level fee_cost.")

    # ── settlement-level comparison ──────────────────────────────────────
    # group fills by ticker, sum formula fee per ticker split by role,
    # compare to settlement fee_cost
    fills_by_ticker = defaultdict(list)
    for r in rows:
        fills_by_ticker[r[0]].append(r)

    print("\n=== SETTLEMENT-LEVEL FEE COMPARISON ===")
    n_match_all_taker = n_match_taker_only = n_match_zero = n_other = 0
    mismatches = []
    pure = {"maker": [0, 0.0, 0.0], "taker": [0, 0.0, 0.0]}  # tickers, actual, formula
    for s in settlements:
        t = s.get("ticker", "")
        fee_actual_c = None
        for k in ("fee_cost", "fee_cost_dollars", "fee"):
            if k in s:
                d = to_dollars(s[k])
                if d is not None:
                    fee_actual_c = d * 100.0
                    break
        if fee_actual_c is None or t not in fills_by_ticker:
            continue
        frows = fills_by_ticker[t]
        form_all = sum(r[6] for r in frows)
        form_taker = sum(r[6] for r in frows if r[1] == "taker")
        roles = set(r[1] for r in frows)
        if abs(fee_actual_c - form_all) < 0.5:
            n_match_all_taker += 1
        elif abs(fee_actual_c - form_taker) < 0.5 and form_taker != form_all:
            n_match_taker_only += 1
        elif fee_actual_c == 0 and form_all > 0:
            n_match_zero += 1
        else:
            n_other += 1
            if len(mismatches) < 15:
                mismatches.append((t, roles, fee_actual_c, form_all, form_taker))
        # pure-role tickers are the clean natural experiment
        if roles == {"maker"}:
            pure["maker"][0] += 1
            pure["maker"][1] += fee_actual_c
            pure["maker"][2] += form_all
        elif roles == {"taker"}:
            pure["taker"][0] += 1
            pure["taker"][1] += fee_actual_c
            pure["taker"][2] += form_all

    total = n_match_all_taker + n_match_taker_only + n_match_zero + n_other
    print(f"settlements joined to fills: {total}")
    print(f"  fee == formula on ALL fills (maker+taker charged): {n_match_all_taker}")
    print(f"  fee == formula on TAKER fills only (maker free):   {n_match_taker_only}")
    print(f"  fee == 0 despite fills:                            {n_match_zero}")
    print(f"  other/mismatch:                                    {n_other}")

    print("\nPure-role tickers (every fill same role):")
    for role, (n, act, form) in pure.items():
        if n:
            print(f"  all-{role:5s}: {n:4d} tickers  actual=${act/100:.2f}  "
                  f"taker-formula=${form/100:.2f}  ratio={act/form if form else float('nan'):.3f}")

    if mismatches:
        print("\nSample mismatches (ticker, roles, actual_c, formula_all_c, formula_taker_c):")
        for m in mismatches:
            print(f"  {m}")

    # ── per-series-aware fee model ────────────────────────────────────────
    # series metadata exposes fee_multiplier and fee_type
    # (quadratic vs quadratic_with_maker_fees). Test:
    #   expected = ceil_4dp(fee_multiplier * 0.07 * P * (1-P) * C) for takers
    #   makers: 0 on 'quadratic', formula on 'quadratic_with_maker_fees'?
    series_info = {}
    for s in sorted({r[0].split("-")[0] for r in rows if r[0]}):
        try:
            data = client._get(f"/series/{s}")
            sd = data.get("series", data)
            series_info[s] = {
                "mult": float(sd.get("fee_multiplier", 1)),
                "type": sd.get("fee_type", "?"),
            }
        except Exception:
            series_info[s] = {"mult": 1.0, "type": "?"}

    def expected_fee_c(px, count, mult):
        p = px / 100.0
        return math.ceil(mult * 0.07 * p * (1 - p) * count * 10000) / 100.0

    print("\n=== SERIES-AWARE FEE MODEL: ceil_4dp(mult * 0.07 * P(1-P) * C) ===")
    from collections import defaultdict as dd
    stats = dd(lambda: [0, 0, 0, 0.0, 0.0])  # n, match_formula, match_zero, act, exp
    maker_nonzero = []
    ts_mismatch = []
    for ticker, role, _, count, px, fee_a, _ in rows:
        if fee_a is None:
            continue
        s = ticker.split("-")[0]
        info = series_info.get(s, {"mult": 1.0, "type": "?"})
        exp = expected_fee_c(px, count, info["mult"])
        key = (info["type"], role)
        st = stats[key]
        st[0] += 1
        st[3] += fee_a
        st[4] += exp
        if abs(fee_a - exp) < 0.005:
            st[1] += 1
        elif fee_a == 0:
            st[2] += 1
        else:
            ts_mismatch.append((ticker, role, count, px, fee_a, exp, info))
        if role == "maker" and fee_a > 0:
            maker_nonzero.append((ticker, count, px, fee_a, exp, info))
    for (ftype, role), (n, mf, mz, act, exp) in sorted(stats.items()):
        print(f"  fee_type={ftype:28s} role={role:6s} n={n:4d}  "
              f"match_formula={mf:4d}  fee_zero={mz:4d}  "
              f"actual=${act/100:.2f}  formula=${exp/100:.2f}")
    if maker_nonzero:
        print("\nMaker fills with NONZERO fee:")
        for m in maker_nonzero:
            print(f"  {m}")
    if ts_mismatch:
        print(f"\nUnexplained fills ({len(ts_mismatch)}), sample:")
        for m in ts_mismatch[:12]:
            print(f"  {m}")

    # ── ratio analysis: fee_actual / (0.07 * P(1-P) * C), by role & date ──
    # reveals the effective multiplier structure empirically (float counts)
    print("\n=== FEE / BASE-FORMULA RATIO BY ROLE AND DATE ===")
    print("base = 0.07 * P * (1-P) * count_fp  (no rounding)")
    by_key = dd(list)
    fill_dates = {}
    for f in fills:
        fill_dates[f.get("fill_id")] = (f.get("created_time") or "")[:10]
    for f in fills:
        ticker = f.get("ticker", "")
        side = f.get("side", "")
        cnt = float(f.get("count_fp", 0) or 0)
        yes_c = to_cents(f.get("yes_price_dollars"))
        no_c = to_cents(f.get("no_price_dollars"))
        px = yes_c if side == "yes" else no_c
        fee = to_dollars(f.get("fee_cost")) or 0.0
        if cnt <= 0 or px <= 0:
            continue
        p = px / 100.0
        base = 0.07 * p * (1 - p) * cnt
        if base <= 0:
            continue
        role = "taker" if f.get("is_taker") else "maker"
        date = (f.get("created_time") or "")[:10]
        s = ticker.split("-")[0]
        info = series_info.get(s, {})
        by_key[(role, round(fee / base, 3) if base else 0)].append((date, s))
    # bucket ratios
    ratio_buckets = dd(lambda: Counter())
    for (role, ratio), items in by_key.items():
        for b, lo, hi in [("~0 (free)", -0.01, 0.02), ("~0.25", 0.2, 0.3),
                          ("~0.5", 0.4, 0.6), ("~1.0", 0.9, 1.15),
                          ("other", None, None)]:
            if lo is not None and lo <= ratio <= hi:
                ratio_buckets[(role, b)].update(d for d, _ in items)
                break
        else:
            ratio_buckets[(role, "other")].update(d for d, _ in items)
    for (role, b), dates in sorted(ratio_buckets.items()):
        n = sum(dates.values())
        drange = f"{min(dates)}..{max(dates)}" if dates else ""
        print(f"  {role:6s} ratio {b:10s} n={n:4d}  dates {drange}")

    # date × role effective-multiplier table (median ratio)
    print("\nMedian fee/base ratio by (date, role):")
    med = dd(list)
    for f in fills:
        cnt = float(f.get("count_fp", 0) or 0)
        side = f.get("side", "")
        px = to_cents(f.get("yes_price_dollars")) if side == "yes" else to_cents(f.get("no_price_dollars"))
        fee = to_dollars(f.get("fee_cost")) or 0.0
        if cnt <= 0 or px <= 0:
            continue
        p = px / 100.0
        base = 0.07 * p * (1 - p) * cnt
        role = "taker" if f.get("is_taker") else "maker"
        s = f.get("ticker", "").split("-")[0]
        ftype = series_info.get(s, {}).get("type", "?")
        med[((f.get("created_time") or "")[:10], role, ftype)].append(fee / base)
    for (date, role, ftype), vals in sorted(med.items()):
        vals.sort()
        m = vals[len(vals) // 2]
        print(f"  {date}  {role:6s} {ftype:28s} n={len(vals):3d}  median_ratio={m:.3f}")

    # ── series fee schedule (read-only public endpoint) ──────────────────
    series = sorted({r[0].split("-")[0] for r in rows if r[0]})
    print("\n=== SERIES FEE_CHANGES ===")
    for s in series:
        try:
            data = client._get(f"/series/{s}/fee_changes")
            print(f"  {s}: {json.dumps(data, default=str)}")
        except Exception as e:
            print(f"  {s}: error {e}")
    # also fetch series metadata for fee fields
    for s in series:
        try:
            data = client._get(f"/series/{s}")
            sd = data.get("series", data)
            fee_keys = {k: v for k, v in sd.items() if "fee" in k.lower()}
            print(f"  {s} series fee fields: {fee_keys}")
        except Exception as e:
            print(f"  {s}: series lookup error {e}")


if __name__ == "__main__":
    main()
