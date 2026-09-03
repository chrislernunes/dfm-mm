# DFM Market-Making


**Universe:** EMAAR UH Equity, EMAARDEV UH Equity
**Period:** 1–30 September 2025
**Sessions:** 21 per stock
**Rows processed:** 181,151

## Strategy

Each stock-day is simulated independently.

* Start quoting at **10:02:00**.
* Quote **10,000 shares** at the best bid and **10,000 shares** at the best ask.
* A trade at our quoted price generates a fill using:

```text
fill = min(outstanding_order,
           trade_size * (trade_size / displayed_level_size))
```

* `displayed_level_size` is the best bid/ask size immediately before the trade.
* Each fill schedules a refill to **10,000 shares** exactly **60 seconds** later.
* Risk-increasing fills are constrained by an absolute exposure limit of **AED 1.5m**. A fill is truncated if necessary.
* Residual inventory is liquidated at the day's last trade price.
* Cash and inventory reset between stock-days.

## Trading window

The primary run uses **10:02:00–14:45:00** for quote-driven fills.

At 14:45 the continuous two-sided book ends and the data moves into the closing/auction phase. Invalid bid/ask quotes are not used for fills.

A sensitivity run with `--session-end 15:00:00` was also performed. Total P&L is unchanged for both stocks because trades after 14:45 occur at the eventual liquidation price.

## Data handling

* `BID` and `ASK` rows with price `<= 0` are treated as invalid quotes.
* Same-timestamp events are processed in source row order.
* Share quantities are floored to whole shares.
* A locked market (`bid == ask == trade price`) is left unresolved because the data does not identify the aggressor side.
* Each stock-day is processed independently.
* P&L is reconciled against cash and ending inventory at the liquidation price.

## Baseline results

### Before fees

| Stock              | Total P&L (AED) | Avg Daily P&L | Profitable Days |  Fills |
| ------------------ | --------------: | ------------: | --------------: | -----: |
| EMAAR UH Equity    |  **+17,811.40** |       +848.16 |         10 / 21 | 19,179 |
| EMAARDEV UH Equity |  **-94,003.55** |     -4,476.36 |         13 / 21 |  9,523 |

### After fees

Using the DFM fee schedule from Task 2:

| Stock              | Variable Fee |  Flat Fee | Total Fees |  P&L After Fees |
| ------------------ | -----------: | --------: | ---------: | --------------: |
| EMAAR UH Equity    |   358,288.29 | 60,795.00 | 419,083.29 | **-401,271.89** |
| EMAARDEV UH Equity |   184,375.46 | 33,505.50 | 217,880.96 | **-311,884.51** |


## Experimental inventory skew

A simple inventory-control variant is included in `skew_variant.py`.

Quote size is reduced on the side that increases inventory and increased on the side that reduces it. The underlying event loop and fill mechanics remain unchanged.

| Stock              | Baseline P&L |       Skew P&L | Baseline Max DD |    Skew Max DD |
| ------------------ | -----------: | -------------: | --------------: | -------------: |
| EMAAR UH Equity    |   +17,811.40 | **+55,545.75** |      -24,620.45 |  **-3,281.05** |
| EMAARDEV UH Equity |   -94,003.55 | **-37,783.25** |      -72,804.90 | **-28,428.95** |

This is an experimental variant rather than an optimized strategy. No parameter search or out-of-sample selection was performed.


`validation_report.txt` contains data-quality, fill, refill, and exposure diagnostics.

## Running

Install dependencies:

```bash
pip install pandas numpy openpyxl
```

Run the primary backtest:

```bash
python backtest.py --input "File 1.xlsx" --output results
```

Run the 15:00 sensitivity:

```bash
python backtest.py --input "File 1.xlsx" --output results_1500 --session-end 15:00:00
```

Run the test suite:

```bash
pytest tests/test_backtest.py -v
```

