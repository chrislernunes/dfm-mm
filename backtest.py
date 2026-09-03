
# Quotes 10,000 shares at the best bid and best ask from 10:02:00 to
# 14:45:00. Each fill schedules a refill back to 10,000 on that side 60s
# later. Net exposure is capped at AED 1.5m; a fill that would breach it is
# truncated, not rejected. At day close, residual inventory is marked at the
# last trade price. Cash and inventory reset to zero each day.

# Row-by-row event loop, no vectorization of the strategy logic -- see
# README.md for methodology.


from __future__ import annotations

import argparse
import heapq
import math
import os
import sys
from dataclasses import dataclass
from datetime import time as dtime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

STRATEGY_START: dtime = dtime(10, 2, 0)
STRATEGY_END: dtime = dtime(14, 45, 0)      # no fills or refills at/after this, tested 15:00pm exit further
QUOTE_SIZE: float = 10_000.0                # target displayed size per side
REFILL_SECONDS: int = 60
EXPOSURE_LIMIT_AED: float = 1_500_000.0     # max abs(position * price)

PRICE_TOL: float = 1e-6   # price-matching tolerance (tick size is 0.01-0.05)
QTY_EPS: float = 1e-6     # epsilon before flooring fill quantities, to absorb float noise

REQUIRED_COLUMNS = ["Dates", "Type", "Price", "Size"]


# ---- data loading -----------------------------------------------------

def _find_header_row(path: str, sheet_name: str, max_scan_rows: int = 8) -> int:
    """Row index of the real header (Dates, Type, Price, Size).

    The sheet has a two-row preamble above it (a ticker title, then an
    "Ask,Trade,Bid" legend), so the header isn't row 0 -- scan for it
    rather than hard-code the offset.
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=max_scan_rows)
    wanted = set(REQUIRED_COLUMNS)
    for i in range(len(raw)):
        row_values = set(str(v).strip() for v in raw.iloc[i].tolist())
        if wanted <= row_values:
            return i
    raise ValueError(
        f"Could not locate a header row containing {REQUIRED_COLUMNS} "
        f"in the first {max_scan_rows} rows of sheet {sheet_name!r}."
    )


def load_workbook(path: str) -> Dict[str, pd.DataFrame]:
    """Load every sheet as one stock's event stream.

    Columns: Dates, Type, Price, Size, plus orig_row_seq / trade_date /
    time_of_day. orig_row_seq is the original row order, captured before
    any sorting -- the file has no sequence-number column, so this is the
    tie-break for same-timestamp events.
    """
    xl = pd.ExcelFile(path)
    out: Dict[str, pd.DataFrame] = {}
    for sheet_name in xl.sheet_names:
        header_row = _find_header_row(path, sheet_name)
        df = xl.parse(sheet_name, header=header_row)
        df.columns = [str(c).strip() for c in df.columns]

        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Sheet {sheet_name!r} is missing expected columns {missing}. "
                f"Found columns: {df.columns.tolist()}"
            )

        df = df[REQUIRED_COLUMNS].copy()
        df["orig_row_seq"] = np.arange(len(df))  # capture before any row is dropped

        df["Dates"] = pd.to_datetime(df["Dates"], errors="coerce")
        df["Type"] = df["Type"].astype(str).str.strip().str.upper()
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df["Size"] = pd.to_numeric(df["Size"], errors="coerce")

        n_before = len(df)
        bad_mask = df["Dates"].isna() | df["Price"].isna() | df["Size"].isna()
        n_bad = int(bad_mask.sum())
        if n_bad:
            # None observed in the provided file; drop and report rather than guess.
            print(
                f"[load_workbook] WARNING sheet {sheet_name!r}: dropping {n_bad} "
                f"row(s) with unparseable Dates/Price/Size (of {n_before}).",
                file=sys.stderr,
            )
        df = df.loc[~bad_mask].copy()

        df["trade_date"] = df["Dates"].dt.date
        df["time_of_day"] = df["Dates"].dt.time
        out[sheet_name.strip()] = df
    return out


# ---- core fill / exposure math (unit-tested directly) -----------------

def floor_shares(x: float, eps: float = QTY_EPS) -> int:
    """Floor a computed share quantity to an integer.

    `eps` absorbs float noise (e.g. 15000*(15000/50000) evaluates to
    4499.999999999998, not 4500) without rounding a genuinely fractional
    result upward -- we never credit more shares than the formula implies.
    """
    return math.floor(x + eps)


def compute_raw_fill(order_outstanding: float, trade_size: float, level_size: Optional[float]) -> float:
    """fill = min(order_outstanding, trade_size * (trade_size / level_size))

    The spec's literal formula, not conventional pro-rata. Returns 0.0 if
    any input is missing or non-positive (no fill against an undefined
    book level).
    """
    if order_outstanding is None or order_outstanding <= 0:
        return 0.0
    if trade_size is None or trade_size <= 0:
        return 0.0
    if level_size is None or level_size <= 0:
        return 0.0
    raw = trade_size * (trade_size / level_size)
    return min(order_outstanding, raw)


def max_buy_qty_for_exposure(position: float, price: float, limit_aed: float) -> float:
    """Largest additional BUY qty such that abs(position + qty) * price <= limit_aed."""
    if price is None or price <= 0:
        return 0.0
    return (limit_aed / price) - position


def max_sell_qty_for_exposure(position: float, price: float, limit_aed: float) -> float:
    """Largest additional SELL qty such that abs(position - qty) * price <= limit_aed."""
    if price is None or price <= 0:
        return 0.0
    return position + (limit_aed / price)


def exposure_capped_fill(raw_fill_shares: int, position: float, price: float, side: str, limit_aed: float) -> int:
    """Truncate (never reject) a fill so the post-fill position stays within the AED limit."""
    if raw_fill_shares <= 0:
        return 0
    if side == "buy":
        max_qty = max_buy_qty_for_exposure(position, price, limit_aed)
    elif side == "sell":
        max_qty = max_sell_qty_for_exposure(position, price, limit_aed)
    else:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    max_qty_int = max(0, floor_shares(max_qty))
    return int(min(raw_fill_shares, max_qty_int))


# ---- state -----------------------------------------------------------

@dataclass
class MarketState:
    # Latest known top-of-book. None on a side means no valid quote right now.
    bid_price: Optional[float] = None
    bid_size: Optional[float] = None
    ask_price: Optional[float] = None
    ask_size: Optional[float] = None


@dataclass
class StrategyBook:
    # Outstanding orders, position, and cash for one (stock, day).
    outstanding_bid: float = 0.0
    outstanding_ask: float = 0.0
    initialized_bid: bool = False
    initialized_ask: bool = False

    position: float = 0.0   # +long, -short
    cash: float = 0.0

    buy_qty: float = 0.0
    sell_qty: float = 0.0
    n_buy_fills: int = 0
    n_sell_fills: int = 0
    gross_traded_value: float = 0.0
    max_long_inventory: float = 0.0
    max_short_inventory: float = 0.0    # signed, <= 0
    max_abs_exposure_aed: float = 0.0


# ---- per-day event loop -------------------------------------------------

class DaySimulator:
    """Runs the strategy over one (stock, date) slice of events, sorted by (Dates, orig_row_seq)."""

    def __init__(self, stock: str, date, events: pd.DataFrame, session_end: dtime = STRATEGY_END):
        self.stock = stock
        self.date = date
        self.events = events
        self.session_end = session_end  # overridable per-run for sensitivity checks; see README Sec. 5
        self.market = MarketState()
        self.book = StrategyBook()
        self.fill_rows: List[dict] = []
        self.last_trade_price: Optional[float] = None
        self.refill_heap: List[Tuple[object, int, str]] = []  # (refill_datetime, seq, side)
        self._refill_seq = 0
        self.validation = {
            "rows_processed": 0,
            "bid_rows": 0,
            "ask_rows": 0,
            "trade_rows": 0,
            "unrecognized_type_rows": 0,
            "zero_or_invalid_price_bidask_rows": 0,
            "trades_in_live_window": 0,
            "trades_matched_bid_side": 0,
            "trades_matched_ask_side": 0,
            "trades_matched_neither_side": 0,
            "locked_market_prints_unresolved": 0,
            "fills_generated": 0,
            "refills_applied": 0,
            "refills_skipped_after_cutoff": 0,
            "cap_violations": 0,
            "exposure_drift_benign": 0,
        }

    def run(self) -> Tuple[dict, List[dict], dict]:
        rows = list(self.events.itertuples(index=False))
        n = len(rows)
        idx = 0
        crossed_start = False

        while idx < n or self.refill_heap:
            next_market_dt = rows[idx].Dates if idx < n else None
            next_refill = self.refill_heap[0] if self.refill_heap else None
            next_refill_dt = next_refill[0] if next_refill is not None else None

            # A market event at the exact same timestamp as a due refill is
            # processed first; the refill fires immediately after (see README).
            if next_refill_dt is not None and (next_market_dt is None or next_refill_dt < next_market_dt):
                heapq.heappop(self.refill_heap)
                self._apply_refill(next_refill_dt, next_refill[2])
                continue

            row = rows[idx]
            if not crossed_start and row.Dates.time() >= STRATEGY_START:
                self._attempt_initialize()
                crossed_start = True
            self._process_event(row)
            idx += 1

        pnl_record = self._finalize_day()
        return pnl_record, self.fill_rows, self.validation

    def _attempt_initialize(self) -> None:
        """Called once, right before the first event at/after 10:02:00, using
        whatever market state is already known. A side with no quote yet is
        picked up by the lazy fallback in _update_side once one arrives.
        """
        if self.market.bid_price is not None:
            self.book.outstanding_bid = QUOTE_SIZE
            self.book.initialized_bid = True
        if self.market.ask_price is not None:
            self.book.outstanding_ask = QUOTE_SIZE
            self.book.initialized_ask = True

    def _process_event(self, row) -> None:
        self.validation["rows_processed"] += 1
        t = row.Dates.time()
        in_live_window = STRATEGY_START <= t < self.session_end

        if row.Type == "BID":
            self.validation["bid_rows"] += 1
            self._update_side("bid", row.Price, row.Size, in_live_window)
        elif row.Type == "ASK":
            self.validation["ask_rows"] += 1
            self._update_side("ask", row.Price, row.Size, in_live_window)
        elif row.Type == "TRADE":
            self.validation["trade_rows"] += 1
            self._process_trade(row, in_live_window)
        else:
            self.validation["unrecognized_type_rows"] += 1

    def _update_side(self, side: str, price, size, in_live_window: bool) -> None:
        valid = price is not None and price > 0
        if valid:
            if side == "bid":
                self.market.bid_price = float(price)
                self.market.bid_size = float(size) if (size is not None and size > 0) else 0.0
                if in_live_window and not self.book.initialized_bid:
                    self.book.outstanding_bid = QUOTE_SIZE
                    self.book.initialized_bid = True
            else:
                self.market.ask_price = float(price)
                self.market.ask_size = float(size) if (size is not None and size > 0) else 0.0
                if in_live_window and not self.book.initialized_ask:
                    self.book.outstanding_ask = QUOTE_SIZE
                    self.book.initialized_ask = True
        else:
            # price <= 0: no valid quote on this side. Only occurs pre-open
            # and during the closing auction in this dataset -- never intraday.
            self.validation["zero_or_invalid_price_bidask_rows"] += 1
            if side == "bid":
                self.market.bid_price = None
                self.market.bid_size = None
            else:
                self.market.ask_price = None
                self.market.ask_size = None

    def _process_trade(self, row, in_live_window: bool) -> None:
        price = row.Price
        size = row.Size
        valid_trade = price is not None and price > 0

        if valid_trade:
            # Tracked independent of the live window: EOD liquidation uses
            # the day's actual last trade, which here is always the
            # 15:00:20 zero-size closing-auction print.
            self.last_trade_price = float(price)

        if not in_live_window or not valid_trade:
            return

        self.validation["trades_in_live_window"] += 1

        bid_eligible = (
            self.book.initialized_bid and self.market.bid_price is not None
            and math.isclose(price, self.market.bid_price, abs_tol=PRICE_TOL)
        )
        ask_eligible = (
            self.book.initialized_ask and self.market.ask_price is not None
            and math.isclose(price, self.market.ask_price, abs_tol=PRICE_TOL)
        )

        if bid_eligible and ask_eligible:
            # Locked market (bid == ask == trade price). A single TRADE row is
            # one execution with one direction -- it can be a fill against our
            # bid OR our ask, never both at once. The data has no aggressor-side
            # field to tell us which, so rather than guess (arbitrarily, or via
            # a heuristic dressed up as principled), we treat the print as
            # unresolved and fill neither side. Affects 3 prints out of 181,151
            # rows -- see README Sec. 9.
            self.validation["locked_market_prints_unresolved"] += 1
            return

        matched = False

        if bid_eligible:
            matched = True
            self.validation["trades_matched_bid_side"] += 1
            raw = compute_raw_fill(self.book.outstanding_bid, size, self.market.bid_size)
            fill = floor_shares(raw)
            if fill > 0:
                fill = exposure_capped_fill(fill, self.book.position, price, "buy", EXPOSURE_LIMIT_AED)
            if fill > 0:
                self._execute_fill(row.Dates, "BUY", price, size, self.market.bid_size, fill)

        if ask_eligible:
            matched = True
            self.validation["trades_matched_ask_side"] += 1
            raw = compute_raw_fill(self.book.outstanding_ask, size, self.market.ask_size)
            fill = floor_shares(raw)
            if fill > 0:
                fill = exposure_capped_fill(fill, self.book.position, price, "sell", EXPOSURE_LIMIT_AED)
            if fill > 0:
                self._execute_fill(row.Dates, "SELL", price, size, self.market.ask_size, fill)

        if not matched:
            self.validation["trades_matched_neither_side"] += 1

    def _execute_fill(self, dt, side: str, price: float, market_trade_size, market_level_size, fill: int) -> None:
        if side == "BUY":
            before = self.book.outstanding_bid
            self.book.cash -= fill * price
            self.book.position += fill
            self.book.outstanding_bid -= fill
            after = self.book.outstanding_bid
            self.book.buy_qty += fill
            self.book.n_buy_fills += 1
            refill_side = "bid"
        else:
            before = self.book.outstanding_ask
            self.book.cash += fill * price
            self.book.position -= fill
            self.book.outstanding_ask -= fill
            after = self.book.outstanding_ask
            self.book.sell_qty += fill
            self.book.n_sell_fills += 1
            refill_side = "ask"

        self.book.gross_traded_value += fill * price
        self.book.max_long_inventory = max(self.book.max_long_inventory, self.book.position)
        self.book.max_short_inventory = min(self.book.max_short_inventory, self.book.position)
        exposure_now = abs(self.book.position) * price
        self.book.max_abs_exposure_aed = max(self.book.max_abs_exposure_aed, exposure_now)

        # Hard constraint: a fill that increases abs(position) can never push
        # exposure past the limit -- guaranteed by exposure_capped_fill().
        # A fill that reduces abs(position) can still show exposure > limit
        # here; that's temporary mark-to-market drift from price movement
        # while cutting inventory, not a cap breach. Tracked separately.
        pos_before = self.book.position - fill if side == "BUY" else self.book.position + fill
        risk_increasing = abs(self.book.position) > abs(pos_before)
        if exposure_now > EXPOSURE_LIMIT_AED + 1.0:  # 1 AED float tolerance
            if risk_increasing:
                self.validation["cap_violations"] += 1
            else:
                self.validation["exposure_drift_benign"] += 1

        refill_dt = dt + timedelta(seconds=REFILL_SECONDS)
        self._refill_seq += 1
        heapq.heappush(self.refill_heap, (refill_dt, self._refill_seq, refill_side))

        self.fill_rows.append({
            "date": self.date,
            "stock": self.stock,
            "timestamp": dt,
            "side": side,
            "price": price,
            "market_trade_size": market_trade_size,
            "market_level_size": market_level_size,
            "our_order_before": before,
            "simulated_fill": fill,
            "our_order_after": after,
            "inventory_after": self.book.position,
            "cash_after": self.book.cash,
            "scheduled_refill_time": refill_dt,
        })
        self.validation["fills_generated"] += 1

    def _apply_refill(self, refill_dt, side: str) -> None:
        # A refill due at/after 14:45:00 is not applied -- quoting has
        # stopped by then. No P&L effect either way since fills also stop
        # at 14:45:00, but kept for a consistent state trace.
        if refill_dt.time() >= self.session_end:
            self.validation["refills_skipped_after_cutoff"] += 1
            return
        if side == "bid":
            self.book.outstanding_bid = QUOTE_SIZE
        else:
            self.book.outstanding_ask = QUOTE_SIZE
        self.validation["refills_applied"] += 1

    def _finalize_day(self) -> dict:
        final_inventory = self.book.position
        cash = round(self.book.cash, 2)
        if self.last_trade_price is not None:
            liquidation_price = self.last_trade_price
            final_pnl = round(cash + final_inventory * liquidation_price, 2)
        else:
            # No valid trade printed all day (never observed in this dataset).
            # Inventory must be 0 in that case -- a position can only arise
            # from a fill, which requires a valid trade -- so PnL is just cash.
            liquidation_price = float("nan")
            final_pnl = cash

        return {
            "date": self.date,
            "stock": self.stock,
            "final_pnl_aed": final_pnl,
            "buy_quantity": self.book.buy_qty,
            "sell_quantity": self.book.sell_qty,
            "number_of_buy_fills": self.book.n_buy_fills,
            "number_of_sell_fills": self.book.n_sell_fills,
            "ending_inventory": final_inventory,
            "max_long_inventory": self.book.max_long_inventory,
            "max_short_inventory": self.book.max_short_inventory,
            "max_absolute_exposure_aed": round(self.book.max_abs_exposure_aed, 2),
            "gross_traded_value": round(self.book.gross_traded_value, 2),
            "total_filled_quantity": self.book.buy_qty + self.book.sell_qty,
            "liquidation_price": liquidation_price,
            "cash_before_liquidation": cash,
        }


# ---- orchestration across the workbook --------------------------------

def _stable_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by (Dates, orig_row_seq).

    No-op on the provided file, which is already time-ordered, but
    guarantees correct same-timestamp tie-breaking on an unsorted workbook.
    """
    return df.sort_values(["Dates", "orig_row_seq"], kind="mergesort").reset_index(drop=True)


def verify_pnl_reconciliation(pnl_df: pd.DataFrame, tol: float = 1e-6) -> None:
    """Check final_pnl_aed == cash_before_liquidation + ending_inventory * liquidation_price
    (or just cash, on a no-trade day) for every row.

    Raises with the offending rows if not. Called on every run, not just in
    tests, so a reconciliation break fails the backtest itself.
    """
    df = pnl_df
    has_price = df["liquidation_price"].notna()
    recon = df["cash_before_liquidation"].copy()
    recon[has_price] = recon[has_price] + df.loc[has_price, "ending_inventory"] * df.loc[has_price, "liquidation_price"]
    diff = (recon - df["final_pnl_aed"]).abs()
    bad = df[diff > tol]
    if not bad.empty:
        cols = ["stock", "date", "final_pnl_aed", "cash_before_liquidation", "ending_inventory", "liquidation_price"]
        raise AssertionError(f"PnL reconciliation failed for {len(bad)} row(s):\n{bad[cols].to_string()}")


def simulate_workbook(input_path: str, session_end: dtime = STRATEGY_END) -> dict:
    """Run the strategy over every stock/day in the workbook.

    session_end defaults to the spec's primary 14:45:00 cutoff; pass a
    later time (e.g. 15:00:00) to run the Sec. 5 sensitivity check against
    the literal "stop at 3pm" wording.
    """
    stocks = load_workbook(input_path)

    pnl_records: List[dict] = []
    fill_rows: List[dict] = []
    validation_totals: Dict[str, int] = {}
    per_day_notes: List[str] = []

    for stock, df in stocks.items():
        df = _stable_sort(df)

        for date, day_df in df.groupby("trade_date", sort=True):
            sim = DaySimulator(stock, date, day_df, session_end=session_end)
            pnl_record, fills, validation = sim.run()
            pnl_records.append(pnl_record)
            fill_rows.extend(fills)
            for k, v in validation.items():
                validation_totals[k] = validation_totals.get(k, 0) + v

            if pd.isna(pnl_record["liquidation_price"]):
                per_day_notes.append(f"{stock} {date}: no valid trade all day, P&L is 0.")
            if validation["cap_violations"] > 0:
                per_day_notes.append(
                    f"{stock} {date}: {validation['cap_violations']} exposure cap violation(s) on a "
                    f"risk-increasing fill -- indicates a bug, should always be 0."
                )
            if validation["exposure_drift_benign"] > 0:
                per_day_notes.append(
                    f"{stock} {date}: mark-to-market exposure briefly above AED 1.5m on "
                    f"{validation['exposure_drift_benign']} risk-reducing fill(s) "
                    f"(max={pnl_record['max_absolute_exposure_aed']:.0f} AED) -- price drift while "
                    f"cutting inventory, not a cap breach; see README."
                )
            if validation["trades_matched_neither_side"] > 0.5 * max(validation["trades_in_live_window"], 1):
                per_day_notes.append(
                    f"{stock} {date}: majority of live-window trades matched neither side "
                    f"({validation['trades_matched_neither_side']}/{validation['trades_in_live_window']})."
                )

    pnl_df = pd.DataFrame(pnl_records).sort_values(["stock", "date"]).reset_index(drop=True)
    verify_pnl_reconciliation(pnl_df)

    fills_df = pd.DataFrame(fill_rows)
    if not fills_df.empty:
        # Stable sort: preserves the true processing order of fills sharing
        # an identical timestamp within a stock (this only groups by
        # stock/time, it doesn't reorder within a group). A non-stable sort
        # would silently scramble same-timestamp fills in the output
        # without affecting the underlying P&L.
        fills_df = fills_df.sort_values(["stock", "timestamp"], kind="mergesort").reset_index(drop=True)

    summary_df = _build_summary(pnl_df)

    return {
        "pnl_df": pnl_df,
        "fills_df": fills_df,
        "summary_df": summary_df,
        "validation_totals": validation_totals,
        "per_day_notes": per_day_notes,
        "stocks": list(stocks.keys()),
        "session_end": session_end,
    }


def _max_drawdown(daily_pnl: pd.Series) -> float:
    """Max peak-to-trough decline of the cumulative daily P&L series (AED)."""
    if daily_pnl.empty:
        return 0.0
    cum = daily_pnl.cumsum()
    drawdown = cum - cum.cummax()
    return float(drawdown.min())


def _build_summary(pnl_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stock, g in pnl_df.groupby("stock"):
        g = g.sort_values("date")
        rows.append({
            "stock": stock,
            "total_pnl_aed": round(g["final_pnl_aed"].sum(), 2),
            "average_daily_pnl_aed": round(g["final_pnl_aed"].mean(), 2),
            "number_of_trading_days": len(g),
            "profitable_days": int((g["final_pnl_aed"] > 0).sum()),
            "losing_days": int((g["final_pnl_aed"] < 0).sum()),
            "flat_days": int((g["final_pnl_aed"] == 0).sum()),
            "total_volume_shares": g["total_filled_quantity"].sum(),
            "total_gross_traded_value_aed": round(g["gross_traded_value"].sum(), 2),
            "max_drawdown_aed": round(_max_drawdown(g["final_pnl_aed"]), 2),
            "total_number_of_fills": int((g["number_of_buy_fills"] + g["number_of_sell_fills"]).sum()),
        })
    return pd.DataFrame(rows)


# ---- validation report --------------------------------------------------

def build_validation_report(result: dict) -> str:
    v = result["validation_totals"]
    lines = []
    lines.append("=" * 78)
    lines.append("DATA VALIDATION REPORT")
    lines.append("=" * 78)
    lines.append(f"Stocks processed: {len(result['stocks'])} -> {result['stocks']}")
    lines.append(f"Trading days processed (stock-day pairs): {len(result['pnl_df'])}")
    lines.append("")
    lines.append(f"Total rows processed:                         {v.get('rows_processed', 0):>10,}")
    lines.append(f"  BID rows:                                   {v.get('bid_rows', 0):>10,}")
    lines.append(f"  ASK rows:                                   {v.get('ask_rows', 0):>10,}")
    lines.append(f"  TRADE rows:                                 {v.get('trade_rows', 0):>10,}")
    lines.append(f"  Unrecognized Type rows:                     {v.get('unrecognized_type_rows', 0):>10,}")
    lines.append("")
    lines.append(f"BID/ASK rows with price<=0 (invalid/no quote): {v.get('zero_or_invalid_price_bidask_rows', 0):>9,}")
    lines.append("  (expected: pre-open warm-up and closing-auction phase only)")
    lines.append("")
    se = result.get("session_end", STRATEGY_END)
    lines.append(f"TRADE rows falling inside the [{STRATEGY_START},{se}) live window: {v.get('trades_in_live_window', 0):>7,}")
    lines.append(f"  ... matched our bid (price==best bid):      {v.get('trades_matched_bid_side', 0):>10,}")
    lines.append(f"  ... matched our ask (price==best ask):      {v.get('trades_matched_ask_side', 0):>10,}")
    lines.append(f"  ... matched neither side:                   {v.get('trades_matched_neither_side', 0):>10,}")
    lines.append(f"  ... locked market, unresolved (see README):  {v.get('locked_market_prints_unresolved', 0):>9,}")
    lines.append("")
    lines.append(f"Simulated fills generated (qty>0 after all caps): {v.get('fills_generated', 0):>7,}")
    lines.append(f"Refills applied:                              {v.get('refills_applied', 0):>10,}")
    lines.append(f"Refills skipped (scheduled at/after {se}): {v.get('refills_skipped_after_cutoff', 0):>9,}")
    lines.append("")
    lines.append("Exposure cap check (every fill, every day):")
    gv = v.get("cap_violations", 0)
    bd = v.get("exposure_drift_benign", 0)
    flag = "  <- unexpected" if gv > 0 else ""
    lines.append(f"  Cap violations on a risk-increasing fill (should be 0): {gv:>5,}{flag}")
    lines.append(f"  Mark-to-market drift above limit on a risk-reducing fill: {bd:>3,}")
    lines.append("    (price moved between fills; the fill itself only ever reduced exposure --")
    lines.append("     see README 'Exposure cap: fill-time vs mark-to-market' for the full explanation)")
    lines.append("")
    lines.append("Rows skipped at load time (unparseable Dates/Price/Size): see stderr warnings above, if any.")
    lines.append("")
    if result["per_day_notes"]:
        lines.append(f"Days flagged with unusual conditions ({len(result['per_day_notes'])}):")
        for note in result["per_day_notes"]:
            lines.append(f"  - {note}")
    else:
        lines.append("Days flagged with unusual conditions: none.")
    lines.append("=" * 78)
    return "\n".join(lines)


# ---- CLI ----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Arqaam market-making event-driven backtest.")
    parser.add_argument("--input", required=True, help='Path to input workbook, e.g. "File 1.xlsx"')
    parser.add_argument("--output", required=True, help='Output directory, e.g. "results/"')
    parser.add_argument("--session-end", default="14:45:00", metavar="HH:MM:SS",
                         help='Session end used for fills/refills (default: 14:45:00, the primary '
                              'result). Pass 15:00:00 to run the literal-spec sensitivity check '
                              'described in README Sec. 5.')
    args = parser.parse_args()
    session_end = dtime.fromisoformat(args.session_end)

    os.makedirs(args.output, exist_ok=True)

    result = simulate_workbook(args.input, session_end=session_end)

    pnl_path = os.path.join(args.output, "daily_pnl.csv")
    fills_path = os.path.join(args.output, "fills.csv")
    summary_path = os.path.join(args.output, "summary.csv")
    validation_path = os.path.join(args.output, "validation_report.txt")

    result["pnl_df"].to_csv(pnl_path, index=False)
    result["fills_df"].to_csv(fills_path, index=False)
    result["summary_df"].to_csv(summary_path, index=False)

    report = build_validation_report(result)
    with open(validation_path, "w") as f:
        f.write(report + "\n")

    print(report)
    print(f"\nWrote: {pnl_path}")
    print(f"Wrote: {fills_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {validation_path}")


if __name__ == "__main__":
    main()