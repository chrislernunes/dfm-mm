# One concrete, minimal inventory-skew variant of the baseline strategy --
# built to answer "what would inventory skew actually do here" with real
# numbers, not just list it as a bullet. Reuses the baseline engine's fill
# formula, exposure cap, and event loop exactly; the ONLY change is that the
# flat 10,000-share quote size is replaced by a size that shrinks on the side
# that would add to inventory and grows on the side that reduces it, scaled
# by how far position already is from zero.

# This is a first pass, not a tuned strategy: one skew scale, chosen to be
# the right order of magnitude for the inventories this book actually builds
# (tens of thousands of shares), not fit to these results. 


from __future__ import annotations

import sys
from datetime import time as dtime

import pandas as pd

from backtest import (
    DaySimulator, QUOTE_SIZE, STRATEGY_START, load_workbook, _stable_sort,
    verify_pnl_reconciliation, _build_summary, _max_drawdown,
)

INV_SCALE = 60_000.0   # shares of position at which skew reaches full strength
SKEW_STRENGTH = 1.0    # 1.0 = size can shrink to 0 on the "wrong" side at full scale


def skewed_size(position: float, side: str) -> float:
    skew = max(-1.0, min(1.0, position / INV_SCALE))  # -1..+1, + = long
    if side == "bid":   # buying adds to a long position -> shrink the bid when long
        return max(0.0, QUOTE_SIZE * (1 - SKEW_STRENGTH * skew))
    else:                # selling adds to a short position -> shrink the ask when short
        return max(0.0, QUOTE_SIZE * (1 + SKEW_STRENGTH * skew))


class DaySimulatorSkewed(DaySimulator):
    def _attempt_initialize(self) -> None:
        if self.market.bid_price is not None:
            self.book.outstanding_bid = skewed_size(self.book.position, "bid")
            self.book.initialized_bid = True
        if self.market.ask_price is not None:
            self.book.outstanding_ask = skewed_size(self.book.position, "ask")
            self.book.initialized_ask = True

    def _update_side(self, side, price, size, in_live_window) -> None:
        valid = price is not None and price > 0
        if valid:
            if side == "bid":
                self.market.bid_price = float(price)
                self.market.bid_size = float(size) if (size is not None and size > 0) else 0.0
                if in_live_window and not self.book.initialized_bid:
                    self.book.outstanding_bid = skewed_size(self.book.position, "bid")
                    self.book.initialized_bid = True
            else:
                self.market.ask_price = float(price)
                self.market.ask_size = float(size) if (size is not None and size > 0) else 0.0
                if in_live_window and not self.book.initialized_ask:
                    self.book.outstanding_ask = skewed_size(self.book.position, "ask")
                    self.book.initialized_ask = True
        else:
            self.validation["zero_or_invalid_price_bidask_rows"] += 1
            if side == "bid":
                self.market.bid_price = None
                self.market.bid_size = None
            else:
                self.market.ask_price = None
                self.market.ask_size = None

    def _apply_refill(self, refill_dt, side: str) -> None:
        if refill_dt.time() >= self.session_end:
            self.validation["refills_skipped_after_cutoff"] += 1
            return
        target = skewed_size(self.book.position, side)
        if side == "bid":
            self.book.outstanding_bid = target
        else:
            self.book.outstanding_ask = target
        self.validation["refills_applied"] += 1


def simulate_skewed(input_path: str) -> dict:
    stocks = load_workbook(input_path)
    pnl_records, fill_rows, validation_totals = [], [], {}
    for stock, df in stocks.items():
        df = _stable_sort(df)
        for date, day_df in df.groupby("trade_date", sort=True):
            sim = DaySimulatorSkewed(stock, date, day_df)
            pnl_record, fills, validation = sim.run()
            pnl_records.append(pnl_record)
            fill_rows.extend(fills)
            for k, v in validation.items():
                validation_totals[k] = validation_totals.get(k, 0) + v
    pnl_df = pd.DataFrame(pnl_records).sort_values(["stock", "date"]).reset_index(drop=True)
    verify_pnl_reconciliation(pnl_df)
    fills_df = pd.DataFrame(fill_rows)
    if not fills_df.empty:
        fills_df = fills_df.sort_values(["stock", "timestamp"], kind="mergesort").reset_index(drop=True)
    summary_df = _build_summary(pnl_df)
    return {"pnl_df": pnl_df, "fills_df": fills_df, "summary_df": summary_df,
            "validation_totals": validation_totals}


if __name__ == "__main__":
    result = simulate_skewed(sys.argv[1] if len(sys.argv) > 1 else "File 1.xlsx")
    print(result["summary_df"].to_string(index=False))
    result["pnl_df"].to_csv("daily_pnl_skewed.csv", index=False)
    result["summary_df"].to_csv("summary_skewed.csv", index=False)
