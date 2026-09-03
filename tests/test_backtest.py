
# Unit tests for the market-making simulator.

# one or more tests per behavior called out in the spec (fill formula,
# refill timing, exposure cap, EOD liquidation, session window, daily reset),
# plus edge cases found while inspecting the real data (locked markets,
# floating-point rounding, multi-day resets) and the reconciliation/invariant
# checks added in the post-implementation audit.

# Run with:  pytest tests/test_backtest.py -v

import math
import os
import sys
from datetime import date as ddate, datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest import (  # noqa: E402
    DaySimulator,
    compute_raw_fill,
    exposure_capped_fill,
    floor_shares,
    max_buy_qty_for_exposure,
    max_sell_qty_for_exposure,
    load_workbook,
    simulate_workbook,
    verify_pnl_reconciliation,
    _stable_sort,
    STRATEGY_START,
    STRATEGY_END,
    QUOTE_SIZE,
    EXPOSURE_LIMIT_AED,
)

DAY = ddate(2025, 9, 1)
UPLOAD_PATH = "File 1.xlsx"


def make_events(rows) -> pd.DataFrame:
   
    # rows: list of (time_str "HH:MM:SS", Type, Price, Size) in the exact
    # intended processing order. Builds a DataFrame matching load_workbook's
    # output schema (Dates, Type, Price, Size, orig_row_seq, trade_date,
    # time_of_day), so DaySimulator sees exactly what the real pipeline
    # would hand it.
   
    records = []
    for i, (t, typ, price, size) in enumerate(rows):
        dt = datetime.combine(DAY, datetime.strptime(t, "%H:%M:%S").time())
        records.append({"Dates": dt, "Type": typ, "Price": float(price), "Size": float(size), "orig_row_seq": i})
    df = pd.DataFrame(records)
    df["trade_date"] = df["Dates"].dt.date
    df["time_of_day"] = df["Dates"].dt.time
    return df


def run_sim(rows):
    # Build events, run one DaySimulator, return (sim, pnl_record, fills, validation).
    events = make_events(rows)
    sim = DaySimulator("TEST", DAY, events)
    pnl_record, fills, validation = sim.run()
    return sim, pnl_record, fills, validation


# -- 1. Exact fill formula --

def test_exact_fill_formula_matches_spec_example():
    # order=10,000 ; trade=15,000 ; level=50,000 -> expected fill = 4,500
    raw = compute_raw_fill(10_000, 15_000, 50_000)
    assert floor_shares(raw) == 4500


def test_fill_formula_is_not_conventional_pro_rata():
    # Conventional pro-rata would give trade*(order/level) = 15000*(10000/50000)=3000.
    # The spec's formula gives trade*(trade/level) capped by order = 4500. Must NOT be 3000.
    raw = compute_raw_fill(10_000, 15_000, 50_000)
    assert floor_shares(raw) != 3000
    assert floor_shares(raw) == 4500


# -- 2/3. Trade at bid -> buy fill; trade at ask -> sell fill --

def test_trade_at_bid_generates_buy_fill():
    rows = [
        ("09:30:00", "BID", 10.00, 100_000),
        ("09:30:00", "ASK", 10.05, 80_000),
        ("10:05:00", "TRADE", 10.00, 1_000),  # matches bid
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 1
    f = fills[0]
    assert f["side"] == "BUY"
    expected_fill = floor_shares(compute_raw_fill(10_000, 1_000, 100_000))
    assert f["simulated_fill"] == expected_fill == 10
    assert sim.book.position == 10
    assert sim.book.cash == pytest.approx(-10 * 10.00)


def test_trade_at_ask_generates_sell_fill():
    rows = [
        ("09:30:00", "BID", 10.00, 100_000),
        ("09:30:00", "ASK", 10.05, 80_000),
        ("10:05:00", "TRADE", 10.05, 800),  # matches ask
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 1
    f = fills[0]
    assert f["side"] == "SELL"
    expected_fill = floor_shares(compute_raw_fill(10_000, 800, 80_000))
    assert f["simulated_fill"] == expected_fill == 8
    assert sim.book.position == -8
    assert sim.book.cash == pytest.approx(8 * 10.05)


# -- 4. Trade away from bid/ask -> no fill --

def test_trade_away_from_quotes_generates_no_fill():
    rows = [
        ("09:30:00", "BID", 10.00, 100_000),
        ("09:30:00", "ASK", 10.05, 80_000),
        ("10:05:00", "TRADE", 9.90, 5_000),  # neither bid nor ask
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 0
    assert sim.book.position == 0
    assert sim.book.cash == 0
    assert val["trades_matched_neither_side"] == 1


# -- 5. Fill uses the PRE-trade market level size --

def test_fill_uses_pre_trade_level_size_not_post_trade():
    rows = [
        ("09:30:00", "BID", 10.00, 100_000),
        ("09:30:00", "ASK", 10.05, 80_000),
        ("10:05:00", "TRADE", 10.00, 1_000),   # must use level=100,000 (pre-existing)
        ("10:05:00", "BID", 10.00, 500_000),   # level jumps AFTER the trade
        ("10:05:01", "TRADE", 10.00, 1_000),   # must use level=500,000 (the new size)
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 2
    fill1 = fills[0]["simulated_fill"]
    fill2 = fills[1]["simulated_fill"]
    assert fill1 == floor_shares(compute_raw_fill(10_000, 1_000, 100_000)) == 10
    assert fill2 == floor_shares(compute_raw_fill(10_000 - fill1, 1_000, 500_000)) == 2
    assert fill1 != fill2  # proves the two trades used genuinely different level sizes


# -- 6. Same-timestamp events preserve source (row) order --

def test_same_timestamp_events_preserve_row_order():
    rows = [
        ("09:30:00", "BID", 10.00, 100_000),
        ("09:30:00", "ASK", 10.05, 80_000),
        ("10:05:00", "TRADE", 10.00, 1_000),   # A: uses level=100,000
        ("10:05:00", "BID", 10.00, 50_000),    # B: same ts as A and C, must apply between them
        ("10:05:00", "TRADE", 10.00, 1_000),   # C: uses level=50,000 because B precedes it in row order
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 2
    fillA, fillC = fills[0]["simulated_fill"], fills[1]["simulated_fill"]
    assert fillA == 10   # 1000*(1000/100000)
    assert fillC == 20   # 1000*(1000/50000), using outstanding=9990 (not binding) and level=50000


def test_deterministic_order_recovers_correct_sequence_even_if_rows_are_shuffled():
    # Rows handed in scrambled order with correct orig_row_seq -- _stable_sort
    # must restore the true processing sequence.
    rows = [
        ("09:30:00", "BID", 10.00, 100_000),
        ("09:30:00", "ASK", 10.05, 80_000),
        ("10:05:00", "TRADE", 10.00, 1_000),   # seq 2 -> A
        ("10:05:00", "BID", 10.00, 50_000),    # seq 3 -> B
        ("10:05:00", "TRADE", 10.00, 1_000),   # seq 4 -> C
    ]
    events = make_events(rows)
    shuffled = events.sample(frac=1.0, random_state=42).reset_index(drop=True)
    restored = _stable_sort(shuffled)
    assert list(restored["orig_row_seq"]) == [0, 1, 2, 3, 4]

    sim = DaySimulator("TEST", DAY, restored)
    _, fills, _ = sim.run()
    assert [f["simulated_fill"] for f in fills] == [10, 20]


# -- 7. Partial fill reduces outstanding quantity (and does not reset it) --

def test_partial_fill_reduces_outstanding_without_resetting():
    rows = [
        ("09:30:00", "BID", 10.00, 1_000_000),
        ("09:30:00", "ASK", 10.05, 80_000),
        ("10:05:00", "TRADE", 10.00, 63_246),  # raw ~4000.06 -> fill 4000
    ]
    sim, pnl, fills, val = run_sim(rows)
    expected = floor_shares(compute_raw_fill(10_000, 63_246, 1_000_000))
    assert expected == 4000
    assert fills[0]["simulated_fill"] == 4000
    # Check the fill record's own snapshot, not sim.book after run()
    # completes: this fill's 60s refill auto-fires before run() returns
    # since no more data is left that day (refills are wall-clock
    # scheduled, not data-dependent -- see the next test).
    assert fills[0]["our_order_before"] == QUOTE_SIZE
    assert fills[0]["our_order_after"] == QUOTE_SIZE - 4000 == 6000


# -- 8/9. Refill exactly 60s later; none before --

def test_no_refill_before_60_seconds():
    rows = [
        ("09:30:00", "BID", 10.00, 1_000_000),
        ("09:30:00", "ASK", 10.05, 80_000),
        ("10:05:00", "TRADE", 10.00, 63_246),   # fill=4000, outstanding->6000, refill@10:06:00
        # huge trade at 59s that would fill the ENTIRE outstanding size, whatever it is right now,
        # letting us read outstanding_bid indirectly through the fill amount:
        ("10:05:59", "TRADE", 10.00, 900_000),  # raw=810,000 >> outstanding either way
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 2
    assert fills[0]["simulated_fill"] == 4000
    assert fills[1]["simulated_fill"] == 6000  # still un-refilled 59s after the first fill


def test_refill_happens_exactly_60_seconds_later():
    rows = [
        ("09:30:00", "BID", 10.00, 1_000_000),
        ("09:30:00", "ASK", 10.05, 80_000),
        ("10:05:00", "TRADE", 10.00, 63_246),   # fill=4000, outstanding->6000, refill@10:06:00
        ("10:06:00", "TRADE", 10.00, 900_000),  # exact tie with refill time
        ("10:06:01", "TRADE", 10.00, 900_000),  # unambiguously after refill time
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 3
    # At the EXACT tie (10:06:00), the same-timestamp market event is processed
    # BEFORE the refill (documented tie-break rule) -> still pre-refill outstanding.
    assert fills[1]["simulated_fill"] == 6000
    # By 10:06:01 the refill has fired -> back to the full 10,000.
    assert fills[2]["simulated_fill"] == 10_000
    # This fill also schedules its own refill (10:07:01), which auto-fires
    # with no more data left -- so again, check the snapshot, not post-run state.
    assert fills[2]["our_order_before"] == 10_000
    assert fills[2]["our_order_after"] == 0


# -- 10/11/12. Exposure cap --

def test_exposure_cap_caps_max_buy_and_sell_quantities():
    # Spec's own worked example: position=100,000, price=14.40 -> ~4,166 more shares
    mq = max_buy_qty_for_exposure(100_000, 14.40, EXPOSURE_LIMIT_AED)
    assert floor_shares(mq) == 4166
    capped = exposure_capped_fill(10_000, 100_000, 14.40, "buy", EXPOSURE_LIMIT_AED)
    assert capped == 4166

    mq_sell = max_sell_qty_for_exposure(-100_000, 14.40, EXPOSURE_LIMIT_AED)
    assert floor_shares(mq_sell) == 4166
    capped_sell = exposure_capped_fill(10_000, -100_000, 14.40, "sell", EXPOSURE_LIMIT_AED)
    assert capped_sell == 4166


def test_long_exposure_never_exceeds_limit_end_to_end():
    # Force many large buy fills at a constant price and confirm the running
    # position * price never breaches AED 1.5m, and the fill is truncated
    # (not rejected outright -- a partial fill still occurs).
    price = 15.00
    rows = [("09:30:00", "BID", price, 10_000_000), ("09:30:00", "ASK", price + 1, 10_000_000)]
    t_minute = 5
    for i in range(30):
        rows.append((f"10:{t_minute:02d}:00", "TRADE", price, 200_000))  # huge trade, always outstanding-binding after refills
        t_minute += 1
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) > 0
    for f in fills:
        assert f["side"] == "BUY"
        exposure_after = abs(f["inventory_after"]) * price
        assert exposure_after <= EXPOSURE_LIMIT_AED + 1e-6
    # the cap must actually have bound at some point (last fill(s) truncated, not full 10,000)
    assert any(f["simulated_fill"] < 10_000 for f in fills)
    assert abs(sim.book.position) * price <= EXPOSURE_LIMIT_AED + 1e-6


def test_short_exposure_never_exceeds_limit_end_to_end():
    price = 15.00
    rows = [("09:30:00", "BID", price - 1, 10_000_000), ("09:30:00", "ASK", price, 10_000_000)]
    t_minute = 5
    for i in range(30):
        rows.append((f"10:{t_minute:02d}:00", "TRADE", price, 200_000))
        t_minute += 1
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) > 0
    for f in fills:
        assert f["side"] == "SELL"
        exposure_after = abs(f["inventory_after"]) * price
        assert exposure_after <= EXPOSURE_LIMIT_AED + 1e-6
    assert any(f["simulated_fill"] < 10_000 for f in fills)
    assert abs(sim.book.position) * price <= EXPOSURE_LIMIT_AED + 1e-6


# -- 13. Inventory / cash accounting --

def test_cash_and_inventory_accounting_is_exact():
    rows = [
        ("09:30:00", "BID", 10.00, 10_000),
        ("09:30:00", "ASK", 10.10, 15_000),
        ("10:05:00", "TRADE", 10.00, 1_000),   # raw=1000*(1000/10000)=100 -> buy fill
        ("10:06:00", "TRADE", 10.10, 1_500),   # raw=1500*(1500/15000)=150 -> sell fill
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 2
    buy_fill = fills[0]["simulated_fill"]
    sell_fill = fills[1]["simulated_fill"]
    assert buy_fill == 100
    assert sell_fill == 150
    expected_cash = -buy_fill * 10.00 + sell_fill * 10.10
    expected_position = buy_fill - sell_fill
    assert sim.book.cash == pytest.approx(expected_cash)
    assert sim.book.position == expected_position


# -- 14. End-of-day liquidation --

def test_end_of_day_liquidation_uses_last_trade_price():
    rows = [
        ("09:30:00", "BID", 10.00, 10_000),
        ("09:30:00", "ASK", 10.10, 1_000_000),
        ("10:05:00", "TRADE", 10.00, 1_000),    # raw=100 -> buy fill inside live window
        ("14:50:00", "TRADE", 11.00, 500),      # after 14:45 -> no fill, but IS the last trade price
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 1
    buy_fill = fills[0]["simulated_fill"]
    assert buy_fill == 100
    expected_pnl = round(-buy_fill * 10.00 + buy_fill * 11.00, 2)
    assert pnl["final_pnl_aed"] == expected_pnl
    assert pnl["liquidation_price"] == 11.00


# -- 15. No fills before 10:02:00 --

def test_no_fills_before_1002():
    rows = [
        ("09:30:00", "BID", 10.00, 100_000),
        ("09:30:00", "ASK", 10.05, 80_000),
        ("09:59:59", "TRADE", 10.00, 5_000),  # would match if strategy were active
        ("10:01:59", "TRADE", 10.00, 5_000),  # still one second before start
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 0
    assert sim.book.position == 0
    assert val["trades_in_live_window"] == 0


# -- 16. Initial quote uses latest valid market state available at 10:02:00 --

def test_initial_quote_uses_latest_state_known_at_1002_with_gap():
    # Mirrors the assignment's own example: last update well before 10:02,
    # nothing exactly at 10:02:00, first live event at 10:02:07.
    rows = [
        ("09:58:00", "BID", 14.40, 208877),
        ("09:58:00", "ASK", 14.45, 119561),
        ("10:02:07", "TRADE", 14.40, 6111),
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 1
    expected = floor_shares(compute_raw_fill(10_000, 6111, 208877))
    assert fills[0]["simulated_fill"] == expected
    assert sim.book.initialized_bid and sim.book.initialized_ask


def test_lazy_initialization_when_no_quote_exists_at_1002():
    # No bid/ask at all before 10:02 -> both sides start un-initialized;
    # the ask becomes tradeable only once a valid ASK update actually arrives.
    rows = [
        ("10:02:00", "TRADE", 14.45, 5_000),   # no quotes yet at all -> cannot fill
        ("10:03:00", "ASK", 14.45, 50_000),    # first valid ask -> lazy-initializes ask side now
        ("10:04:00", "TRADE", 14.45, 1_000),   # should fill now
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 1
    assert fills[0]["side"] == "SELL"
    assert sim.book.initialized_ask is True
    assert sim.book.initialized_bid is False  # never got a valid bid all day


# -- 17. No continuous-trading fills after 14:45:00 --

def test_no_fills_at_or_after_1445():
    rows = [
        ("09:30:00", "BID", 10.00, 100_000),
        ("09:30:00", "ASK", 10.05, 80_000),
        ("14:45:00", "TRADE", 10.00, 5_000),  # exact boundary -> excluded
        ("14:50:00", "TRADE", 10.00, 5_000),  # well after -> excluded
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 0
    assert sim.book.position == 0


def test_fill_still_allowed_one_second_before_1445():
    rows = [
        ("09:30:00", "BID", 10.00, 100_000),
        ("09:30:00", "ASK", 10.05, 80_000),
        ("14:44:59", "TRADE", 10.00, 5_000),
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 1


# -- 18. Daily state resets correctly --

def test_daily_state_resets_between_independent_days():
    day1_rows = [
        ("09:30:00", "BID", 10.00, 1_000_000),
        ("09:30:00", "ASK", 10.05, 1_000_000),
        ("10:05:00", "TRADE", 10.00, 50_000),  # large buy fill, leaves day1 long & with negative cash
    ]
    sim1, pnl1, fills1, _ = run_sim(day1_rows)
    assert sim1.book.position > 0
    assert sim1.book.cash < 0

    # A brand-new DaySimulator for a second day must start from zero,
    # regardless of how day 1 ended.
    day2_rows = [
        ("09:30:00", "BID", 20.00, 10_000),
        ("09:30:00", "ASK", 20.05, 1_000_000),
        ("10:05:00", "TRADE", 20.00, 1_000),  # raw=100
    ]
    sim2, pnl2, fills2, _ = run_sim(day2_rows)
    fill2 = fills2[0]["simulated_fill"]
    # If day1 state had leaked in, cash/position before this fill would be nonzero
    # and the post-fill numbers would not match a clean zero start.
    assert sim2.book.position == fill2
    assert sim2.book.cash == pytest.approx(-fill2 * 20.00)
    assert pnl2["final_pnl_aed"] == round(-fill2 * 20.00 + fill2 * 20.00, 2)  # liquidated at same price = 0 pnl


def test_full_workbook_grouping_resets_across_multiple_days():
    # End-to-end version of the above via the real grouping path used by simulate_workbook.
    rows = []
    day1 = datetime(2025, 9, 1)
    day2 = datetime(2025, 9, 2)
    def add(day, t, typ, price, size):
        rows.append({"Dates": datetime.combine(day.date(), datetime.strptime(t, "%H:%M:%S").time()),
                      "Type": typ, "Price": float(price), "Size": float(size)})
    add(day1, "09:30:00", "BID", 10.00, 1_000_000)
    add(day1, "09:30:00", "ASK", 10.05, 1_000_000)
    add(day1, "10:05:00", "TRADE", 10.00, 50_000)
    add(day2, "09:30:00", "BID", 20.00, 10_000)
    add(day2, "09:30:00", "ASK", 20.05, 1_000_000)
    add(day2, "10:05:00", "TRADE", 20.00, 1_000)  # raw=100
    df = pd.DataFrame(rows)
    df["orig_row_seq"] = np.arange(len(df))
    df["trade_date"] = df["Dates"].dt.date
    df = _stable_sort(df)

    results = {}
    for date, day_df in df.groupby("trade_date", sort=True):
        sim = DaySimulator("TEST", date, day_df)
        pnl, fills, val = sim.run()
        results[date] = (sim, pnl, fills)

    sim2, pnl2, fills2 = results[day2.date()]
    fill2 = fills2[0]["simulated_fill"]
    assert sim2.book.position == fill2  # not contaminated by day1's 50,000-share position


# -- 19. Zero/invalid quote prices do not create bogus fills --

def test_zero_price_bid_does_not_initialize_or_fill():
    rows = [
        ("09:00:00", "BID", 10.00, 500),   # briefly valid
        ("09:30:00", "BID", 0.00, 500),    # invalidated back to "no quote"
        ("09:30:00", "ASK", 10.05, 80_000),
        ("10:05:00", "TRADE", 10.00, 1_000),  # would match the OLD bid price, but bid is now invalid
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert sim.book.initialized_bid is False
    assert len(fills) == 0
    assert val["zero_or_invalid_price_bidask_rows"] == 1


def test_zero_price_trade_never_fills():
    rows = [
        ("09:30:00", "BID", 10.00, 100_000),
        ("09:30:00", "ASK", 10.05, 80_000),
        ("10:05:00", "TRADE", 0.00, 1_000),   # invalid trade price
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 0


# -- 20. Multiple trades at the same timestamp processed independently --

def test_multiple_same_timestamp_trades_processed_independently_no_bidask_between():
    rows = [
        ("09:30:00", "BID", 14.40, 151097),
        ("09:30:00", "ASK", 14.45, 100_000),
        ("10:02:39", "TRADE", 14.40, 23316),
        ("10:02:39", "TRADE", 14.40, 3461),
        ("10:02:39", "TRADE", 14.40, 1500),
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 3
    # Level size (151097) is constant across all three -- no BID update in
    # between -- but outstanding shrinks fill-by-fill, so each fill must be
    # computed independently, not merged into one trade of 23316+3461+1500.
    o0 = QUOTE_SIZE
    f0 = floor_shares(compute_raw_fill(o0, 23316, 151097))
    f1 = floor_shares(compute_raw_fill(o0 - f0, 3461, 151097))
    f2 = floor_shares(compute_raw_fill(o0 - f0 - f1, 1500, 151097))
    assert [f["simulated_fill"] for f in fills] == [f0, f1, f2]
    # All three fills share timestamp 10:02:39, so all three refills land on
    # the same instant, 10:03:39, and auto-fire with no more data that day --
    # each is a flat, idempotent reset to 10,000.
    assert val["refills_applied"] == 3
    assert sim.book.outstanding_bid == QUOTE_SIZE


# -- Additional edge-case tests --

def test_floor_shares_epsilon_handles_float_noise_without_rounding_up_real_fractions():
    assert floor_shares(4499.999999999998) == 4500   # float noise from the spec example -> must round UP to true value
    assert floor_shares(4500.9) == 4500               # genuine fraction -> must NOT round up
    assert floor_shares(0.0) == 0
    assert floor_shares(-0.0000001) == 0


def test_locked_market_single_trade_fills_neither_side():
    # bid == ask == trade price (transient locked market, observed 3 times in
    # the real data). A single TRADE row is one execution with one direction:
    # it cannot be a fill against both our bid and our ask from the same
    # print. With no aggressor-side field to say which side was actually hit,
    # the correct behavior is to leave the print unresolved rather than
    # credit (or guess) either side. See README Sec. 9.
    rows = [
        ("09:30:00", "BID", 10.00, 100_000),
        ("09:30:00", "ASK", 10.00, 90_000),
        ("10:05:00", "TRADE", 10.00, 1_000),
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 0
    assert sim.book.position == 0
    assert sim.book.outstanding_bid == QUOTE_SIZE  # untouched, not partially consumed
    assert sim.book.outstanding_ask == QUOTE_SIZE
    assert val["locked_market_prints_unresolved"] == 1
    assert val["trades_matched_bid_side"] == 0
    assert val["trades_matched_ask_side"] == 0
    assert val["trades_matched_neither_side"] == 0  # distinct from a genuine no-match


def test_refill_restores_to_10000_flat_not_additive():
    # Two separate partial fills before either refill fires; both refills
    # should just drive outstanding back to the flat 10,000 target (not add
    # the filled amounts back cumulatively).
    rows = [
        ("09:30:00", "BID", 10.00, 1_000_000),
        ("09:30:00", "ASK", 10.05, 1_000_000),
        ("10:05:00", "TRADE", 10.00, 63_246),   # fill=4000 -> outstanding=6000, refill@10:06:00
        ("10:05:10", "TRADE", 10.00, 44_721),   # raw~2000 -> fill=2000 -> outstanding=4000, refill@10:06:10
        ("10:06:20", "TRADE", 10.00, 900_000),  # after BOTH refills -> outstanding must be exactly 10,000
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 3
    assert fills[0]["simulated_fill"] == 4000
    assert fills[1]["simulated_fill"] == floor_shares(compute_raw_fill(6000, 44_721, 1_000_000))
    assert fills[-1]["simulated_fill"] == 10_000
    # 3, not 2: each of the first two fills schedules a refill, and the third
    # fill (which fully consumes the restored 10,000) schedules a third that
    # also auto-fires with no more data left.
    assert val["refills_applied"] == 3


def test_refill_scheduled_after_1445_is_not_applied():
    rows = [
        ("09:30:00", "BID", 10.00, 1_000_000),
        ("09:30:00", "ASK", 10.05, 1_000_000),
        ("14:44:30", "TRADE", 10.00, 63_246),  # fill -> refill scheduled 14:45:30, past cutoff
    ]
    sim, pnl, fills, val = run_sim(rows)
    assert len(fills) == 1
    assert val["refills_skipped_after_cutoff"] == 1
    assert sim.book.outstanding_bid == QUOTE_SIZE - fills[0]["simulated_fill"]


def test_reducing_fill_not_blocked_even_if_resulting_mark_to_market_exceeds_limit_at_new_price():
    # Mirrors a real pattern in the data (EMAARDEV, 2025-09-18): a short
    # position compliant at the price that built it is later marked at a
    # new price where its magnitude alone exceeds the limit. Covering part
    # of it is risk-reducing and must never be blocked.
    position = -104_300
    price = 14.40
    assert abs(position) * price > EXPOSURE_LIMIT_AED  # confirms the "already over at this price" setup
    fill = exposure_capped_fill(100, position, price, "buy", EXPOSURE_LIMIT_AED)
    assert fill == 100  # not truncated at all
    new_position = position + fill
    assert abs(new_position) < abs(position)  # genuinely risk-reducing


def test_increasing_fill_is_blocked_when_already_at_limit_at_this_price():
    # Same starting position/price as above, but SELLING (going MORE short)
    # must be fully blocked: we are already past what 14.40 allows.
    position = -104_300
    price = 14.40
    fill = exposure_capped_fill(500, position, price, "sell", EXPOSURE_LIMIT_AED)
    assert fill == 0


def test_end_to_end_benign_drift_is_tracked_separately_from_genuine_violations():
    # Seed a pre-existing short position directly, then feed a trade that
    # covers part of it at a new price. Must (a) fill in full, (b) not
    # count as a genuine violation, (c) count as benign drift.
    rows = [
        ("09:30:00", "BID", 14.35, 1_000_000),
        ("09:30:00", "ASK", 14.45, 1_000_000),  # away from 14.40 so it can't also match the trade below
        ("10:05:00", "TRADE", 14.35, 100),  # tiny warm-up fill just to initialize cleanly
    ]
    sim, pnl, fills, val = run_sim(rows)
    sim.book.position = -104_300.0  # seed as if established earlier at a lower, compliant price
    sim.book.cash = 0.0
    fake_row = type("Row", (), {"Dates": pd.Timestamp("2025-09-01 10:06:00"), "Price": 14.40, "Size": 1_000.0})()
    sim.market.bid_price, sim.market.bid_size = 14.40, 10_000.0  # raw fill = 1000*(1000/10000) = 100
    sim.book.outstanding_bid = 10_000.0
    sim.book.initialized_bid = True
    sim._process_trade(fake_row, in_live_window=True)
    assert sim.book.position > -104_300.0  # covered some of the short (risk-reducing)
    assert val["cap_violations"] == 0
    assert val["exposure_drift_benign"] >= 1


# -- PnL reconciliation and displayed-size invariants --

def test_pnl_reconciles_for_a_multi_fill_multi_day_scenario():
    rows = [
        ("09:30:00", "BID", 10.00, 10_000),
        ("09:30:00", "ASK", 10.10, 15_000),
        ("10:05:00", "TRADE", 10.00, 1_000),
        ("10:06:00", "TRADE", 10.10, 1_500),
        ("14:50:00", "TRADE", 10.50, 500),  # after cutoff: sets liquidation price only
    ]
    _, pnl, _, _ = run_sim(rows)
    recon = pnl["cash_before_liquidation"] + pnl["ending_inventory"] * pnl["liquidation_price"]
    assert recon == pytest.approx(pnl["final_pnl_aed"], abs=1e-6)


def test_verify_pnl_reconciliation_passes_on_consistent_data():
    df = pd.DataFrame([{
        "stock": "X", "date": DAY, "final_pnl_aed": 100.0,
        "cash_before_liquidation": -50.0, "ending_inventory": 10, "liquidation_price": 15.0,
    }])
    verify_pnl_reconciliation(df)  # no exception


def test_verify_pnl_reconciliation_raises_on_a_genuine_break():
    df = pd.DataFrame([{
        "stock": "X", "date": DAY, "final_pnl_aed": 999.0,  # should be 100.0
        "cash_before_liquidation": -50.0, "ending_inventory": 10, "liquidation_price": 15.0,
    }])
    with pytest.raises(AssertionError):
        verify_pnl_reconciliation(df)


def test_verify_pnl_reconciliation_handles_no_trade_day():
    df = pd.DataFrame([{
        "stock": "X", "date": DAY, "final_pnl_aed": 0.0,
        "cash_before_liquidation": 0.0, "ending_inventory": 0, "liquidation_price": float("nan"),
    }])
    verify_pnl_reconciliation(df)  # no exception


def test_outstanding_never_exceeds_quote_size_or_goes_negative():
    rows = [
        ("09:30:00", "BID", 10.00, 1_000_000),
        ("09:30:00", "ASK", 10.05, 1_000_000),
        ("10:05:00", "TRADE", 10.00, 63_246),
        ("10:06:01", "TRADE", 10.00, 900_000),
    ]
    _, _, fills, _ = run_sim(rows)
    for f in fills:
        assert 0 <= f["our_order_before"] <= QUOTE_SIZE
        assert 0 <= f["our_order_after"] <= QUOTE_SIZE


# -- Integration smoke test against the real uploaded workbook --

@pytest.mark.skipif(not os.path.exists(UPLOAD_PATH), reason="real workbook not present in this environment")
def test_load_real_workbook_structure():
    stocks = load_workbook(UPLOAD_PATH)
    assert set(stocks.keys()) == {"EMAAR UH Equity", "EMAARDEV UH Equity"}
    for name, df in stocks.items():
        assert list(df.columns[:4]) == ["Dates", "Type", "Price", "Size"] or \
               set(["Dates", "Type", "Price", "Size"]) <= set(df.columns)
        assert df["Price"].isna().sum() == 0
        assert df["Size"].isna().sum() == 0
        assert set(df["Type"].unique()) <= {"BID", "ASK", "TRADE"}
        assert (df["Price"] >= 0).all()
        assert (df["Size"] >= 0).all()
    assert len(stocks["EMAAR UH Equity"]) == 127511
    assert len(stocks["EMAARDEV UH Equity"]) == 53640


@pytest.fixture(scope="module")
def real_result():
    if not os.path.exists(UPLOAD_PATH):
        pytest.skip("real workbook not present in this environment")
    return simulate_workbook(UPLOAD_PATH)


def test_pnl_reconciliation_across_real_workbook(real_result):
    # simulate_workbook() already runs this internally; re-asserted here so
    # a reconciliation break shows up as a named test failure, not just a
    # crash buried inside a fixture.
    verify_pnl_reconciliation(real_result["pnl_df"])


def test_outstanding_never_exceeds_quote_size_on_real_data(real_result):
    fills = real_result["fills_df"]
    assert (fills["our_order_before"] <= QUOTE_SIZE).all()
    assert (fills["our_order_after"] <= QUOTE_SIZE).all()
    assert (fills["our_order_after"] >= 0).all()


def test_refill_delay_is_always_60_seconds_on_real_data(real_result):
    fills = real_result["fills_df"]
    delta = (pd.to_datetime(fills["scheduled_refill_time"]) - pd.to_datetime(fills["timestamp"])).dt.total_seconds()
    assert (delta == 60).all()


def test_no_fills_outside_live_window_on_real_data(real_result):
    fills = real_result["fills_df"]
    t = pd.to_datetime(fills["timestamp"]).dt.time
    assert ((t >= STRATEGY_START) & (t < STRATEGY_END)).all()


def test_simulate_workbook_fills_csv_preserves_order_for_same_timestamp_fills(tmp_path):
    # fills_df must be sorted with a stable sort: same-timestamp fills need
    # to keep their true processing order in the output CSV. A non-stable
    # sort can silently scramble them (the underlying P&L is unaffected).
    rows = [
        ("09:30:00", "BID", 14.40, 500_000),
        ("09:30:00", "ASK", 14.45, 500_000),
        # three same-timestamp trades against the bid, each individually
        # partial, so the three resulting fills are numerically DISTINCT and
        # therefore only correct if their relative order survives:
        ("10:02:39", "TRADE", 14.40, 23_316),
        ("10:02:39", "TRADE", 14.40, 3_461),
        ("10:02:39", "TRADE", 14.40, 1_500),
    ]
    events = make_events(rows)
    out_xlsx = tmp_path / "synthetic.xlsx"
    with pd.ExcelWriter(out_xlsx) as writer:
        title_and_legend = pd.DataFrame([["SYN UH Equity", None, None, None], ["Ask,Trade,Bid", None, None, None]])
        title_and_legend.to_excel(writer, sheet_name="SYN UH Equity", header=False, index=False)
        events[["Dates", "Type", "Price", "Size"]].to_excel(
            writer, sheet_name="SYN UH Equity", header=True, index=False, startrow=2
        )

    result = simulate_workbook(str(out_xlsx))
    fl = result["fills_df"]
    fl = fl[fl["stock"] == "SYN UH Equity"].reset_index(drop=True)
    assert len(fl) == 3
    o0 = QUOTE_SIZE
    f0 = floor_shares(compute_raw_fill(o0, 23_316, 500_000))
    f1 = floor_shares(compute_raw_fill(o0 - f0, 3_461, 500_000))
    f2 = floor_shares(compute_raw_fill(o0 - f0 - f1, 1_500, 500_000))
    assert list(fl["simulated_fill"]) == [f0, f1, f2]  # true processing order, not scrambled


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
