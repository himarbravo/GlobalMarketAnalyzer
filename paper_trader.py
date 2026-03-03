"""
PAPER TRADER — P4.3
=====================
Daily paper-trading wrapper. Generates signals and logs them to Supabase
without executing real trades. Tracks hypothetical P&L over time.

Usage:
    # Generate signals for today and log to Supabase
    python paper_trader.py

    # Review past paper trades against realized returns
    python paper_trader.py --review

    # Generate for a specific date (backfill)
    python paper_trader.py --date 2026-03-01
"""

import argparse
import logging
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

from db.database_manager import DatabaseManager
from signal_generator import SignalGenerator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

STRATEGY_TAG = "paper_v1"


def run_paper_trade(ref_date: str = None):
    """
    Generate signals for a date and log to Supabase as paper trades.
    Uses strategy="paper_v1" to distinguish from live signals.
    """
    gen = SignalGenerator()
    ref = ref_date or date.today().strftime("%Y-%m-%d")

    logger.info(f"\n{'═' * 70}")
    logger.info(f"  PAPER TRADER — {ref}")
    logger.info(f"{'═' * 70}")

    # Run the full pipeline
    signals = gen.run(reference_date=ref)

    if not signals:
        logger.info("  No signals generated.")
        return

    # Tag all signals as paper trades
    for sig in signals:
        sig["strategy"] = STRATEGY_TAG

    # P5: Get execution mode from signals
    execution_mode = signals[0].get("execution_mode", "alpha") if signals else "alpha"
    logger.info(f"  Execution mode: {execution_mode}")

    # Filter: only save actionable signals (BUY/SELL/WATCH, not HOLD)
    actionable = [s for s in signals if s.get("signal") in ("BUY", "SELL", "WATCH")]

    # P5: Regime-conditional filtering
    REFUGE_TICKERS = {"TLT", "GLD", "SHY", "TIP", "IEF"}
    if execution_mode == "refuge":
        actionable = [s for s in actionable
                      if s["signal"] == "SELL" or s["ticker"] in REFUGE_TICKERS]
        logger.info(f"  REFUGE mode: filtered to {len(actionable)} signals (SELL + refuge)")
    elif execution_mode == "defensive":
        actionable = [s for s in actionable
                      if s["signal"] != "BUY" or s.get("fundamental_score", 0) > 0]
        logger.info(f"  DEFENSIVE mode: filtered to {len(actionable)} signals (quality longs only)")

    if not actionable:
        logger.info("  All signals are HOLD — nothing to log.")
        return

    # Log to Supabase
    db = gen.db
    CHUNK = 400
    total = 0
    for i in range(0, len(actionable), CHUNK):
        chunk = actionable[i:i + CHUNK]
        db.client.table("signals").insert(chunk).execute()
        total += len(chunk)

    logger.info(f"\n  ✓ {total} paper signals logged to Supabase (strategy={STRATEGY_TAG})")

    # Print summary
    _print_summary(signals, gen)


def _print_summary(signals: list, gen: SignalGenerator):
    """Print readable summary of paper trade signals."""
    n_buy = sum(1 for s in signals if s.get("signal") == "BUY")
    n_sell = sum(1 for s in signals if s.get("signal") == "SELL")
    n_watch = sum(1 for s in signals if s.get("signal") == "WATCH")
    n_hold = sum(1 for s in signals if s.get("signal") == "HOLD")

    diag = gen.diagnostics

    logger.info(f"\n  {'─' * 50}")
    logger.info(f"  Date:     {diag.get('date', '?')}")
    logger.info(f"  Assets:   {diag.get('n_assets', '?')}")
    logger.info(f"  s(t):     {diag.get('s', '?')}")
    logger.info(f"  α:        {diag.get('alpha', '?')}")
    logger.info(f"  {'─' * 50}")
    logger.info(f"  BUY:   {n_buy}")
    logger.info(f"  SELL:  {n_sell}")
    logger.info(f"  WATCH: {n_watch}")
    logger.info(f"  HOLD:  {n_hold}")
    logger.info(f"  {'─' * 50}")

    # Top signals
    buys = sorted([s for s in signals if s.get("signal") == "BUY"],
                  key=lambda x: abs(x.get("confidence", 0)), reverse=True)
    sells = sorted([s for s in signals if s.get("signal") == "SELL"],
                   key=lambda x: abs(x.get("confidence", 0)), reverse=True)

    if buys:
        logger.info(f"\n  Top BUY signals:")
        for s in buys[:5]:
            logger.info(f"    {s['ticker']:<8} conf={s.get('confidence', 0):.2f}  "
                        f"regime={s.get('regime', '?')}")

    if sells:
        logger.info(f"\n  Top SELL signals:")
        for s in sells[:5]:
            logger.info(f"    {s['ticker']:<8} conf={s.get('confidence', 0):.2f}  "
                        f"regime={s.get('regime', '?')}")


def review_paper_trades():
    """
    Review past paper trades: compare signals to realized forward returns.
    Scores each signal and computes cumulative statistics.
    """
    db = DatabaseManager()

    logger.info(f"\n{'═' * 70}")
    logger.info(f"  PAPER TRADE REVIEW")
    logger.info(f"{'═' * 70}")

    # Fetch all paper trade signals
    try:
        all_signals = []
        offset = 0
        while True:
            res = (db.client.table("signals")
                   .select("*")
                   .eq("strategy", STRATEGY_TAG)
                   .order("date", desc=False)
                   .range(offset, offset + 999)
                   .execute())
            if not res.data:
                break
            all_signals.extend(res.data)
            if len(res.data) < 1000:
                break
            offset += 1000
    except Exception as e:
        logger.error(f"  ❌ Error fetching signals: {e}")
        return

    if not all_signals:
        logger.info("  No paper trades found in database.")
        logger.info(f"  Run `python paper_trader.py` first to generate signals.")
        return

    df = pd.DataFrame(all_signals)
    df["date"] = pd.to_datetime(df["date"])

    logger.info(f"  Found {len(df)} paper trade signals")
    logger.info(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    logger.info(f"  Tickers: {df['ticker'].nunique()}")

    # ── Score signals against realized returns ──
    results = []
    today = pd.Timestamp(date.today())
    lookback_cutoff = today - timedelta(days=5)  # need 5d forward

    scoreable = df[df["date"] < lookback_cutoff]
    if scoreable.empty:
        logger.info("  No signals old enough to score (need 5+ days).")
        return

    logger.info(f"  Scoring {len(scoreable)} signals (5d+ old)...")

    for _, row in scoreable.iterrows():
        ticker = row["ticker"]
        signal_date = row["date"]
        signal = row.get("signal", "HOLD")

        if signal == "HOLD":
            continue

        # Get forward returns
        try:
            prices = db.get_prices(
                ticker,
                start_date=signal_date.strftime("%Y-%m-%d"),
                end_date=(signal_date + timedelta(days=25)).strftime("%Y-%m-%d"),
            )
        except Exception:
            continue

        if prices.empty or "close" not in prices.columns:
            continue

        close = prices["close"].apply(pd.to_numeric, errors='coerce').dropna()
        if len(close) < 6:
            continue

        entry_price = float(close.iloc[0])
        if entry_price <= 0:
            continue

        # Forward returns
        ret_5d = (float(close.iloc[min(5, len(close) - 1)]) / entry_price - 1) * 100
        ret_20d = None
        if len(close) > 20:
            ret_20d = (float(close.iloc[20]) / entry_price - 1) * 100

        # Was the signal correct?
        direction = 1 if signal == "BUY" else -1 if signal == "SELL" else 0
        correct_5d = (direction * ret_5d) > 0
        correct_20d = (direction * ret_20d) > 0 if ret_20d is not None else None

        results.append({
            "date": signal_date,
            "ticker": ticker,
            "signal": signal,
            "confidence": row.get("confidence", 0),
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "correct_5d": correct_5d,
            "correct_20d": correct_20d,
            "pnl_5d": direction * ret_5d,
        })

    if not results:
        logger.info("  Could not score any signals (missing price data).")
        return

    rdf = pd.DataFrame(results)

    # ── Summary statistics ──
    logger.info(f"\n  {'─' * 60}")
    logger.info(f"  RESULTS — {len(rdf)} scored signals")
    logger.info(f"  {'─' * 60}")

    hit_5d = rdf["correct_5d"].mean()
    scoreable_20d = rdf.dropna(subset=["correct_20d"])
    hit_20d = scoreable_20d["correct_20d"].mean() if len(scoreable_20d) > 0 else None

    avg_pnl_5d = rdf["pnl_5d"].mean()
    total_pnl_5d = rdf["pnl_5d"].sum()

    logger.info(f"  Hit rate (5d):  {hit_5d:.1%} ({int(rdf['correct_5d'].sum())}/{len(rdf)})")
    if hit_20d is not None:
        logger.info(f"  Hit rate (20d): {hit_20d:.1%} ({int(scoreable_20d['correct_20d'].sum())}/{len(scoreable_20d)})")
    logger.info(f"  Avg P&L (5d):   {avg_pnl_5d:+.2f}% per signal")
    logger.info(f"  Total P&L (5d): {total_pnl_5d:+.1f}%")

    # By signal type
    for sig_type in ["BUY", "SELL"]:
        subset = rdf[rdf["signal"] == sig_type]
        if len(subset) > 0:
            logger.info(f"\n  {sig_type} signals ({len(subset)}):")
            logger.info(f"    Hit 5d:  {subset['correct_5d'].mean():.1%}")
            logger.info(f"    Avg P&L: {subset['pnl_5d'].mean():+.2f}%")

    # By date (rolling accuracy)
    logger.info(f"\n  {'─' * 60}")
    logger.info(f"  Rolling accuracy (by date):")
    rdf_sorted = rdf.sort_values("date")
    dates_unique = rdf_sorted["date"].dt.date.unique()
    for d in dates_unique[-10:]:
        day_df = rdf_sorted[rdf_sorted["date"].dt.date == d]
        day_hit = day_df["correct_5d"].mean()
        day_pnl = day_df["pnl_5d"].mean()
        icon = "✅" if day_hit > 0.5 else "❌"
        logger.info(f"    {icon} {d}: hit={day_hit:.0%} pnl={day_pnl:+.2f}% "
                    f"({len(day_df)} signals)")

    # Verdict
    logger.info(f"\n  {'─' * 60}")
    if hit_5d > 0.55:
        logger.info(f"  🏆 Signal quality: STRONG (hit={hit_5d:.1%})")
    elif hit_5d > 0.50:
        logger.info(f"  ✅ Signal quality: POSITIVE (hit={hit_5d:.1%})")
    elif hit_5d > 0.45:
        logger.info(f"  ⚠ Signal quality: MARGINAL (hit={hit_5d:.1%})")
    else:
        logger.info(f"  ❌ Signal quality: POOR (hit={hit_5d:.1%})")
    logger.info(f"{'═' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Paper Trader — P4.3")
    parser.add_argument("--review", action="store_true",
                        help="Review past paper trades vs realized returns")
    parser.add_argument("--date", type=str, default=None,
                        help="Generate signals for a specific date (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.review:
        review_paper_trades()
    else:
        run_paper_trade(ref_date=args.date)


if __name__ == "__main__":
    main()
