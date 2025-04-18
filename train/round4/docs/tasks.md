IMC Prosperity – Round 4 Strategy Brief

1. Round Context

New product: MAGNIFICENT_MACARONS

Position limit: ± 75 lot

Conversion limit: 10 lot/iteration

Key cost drivers (from ConversionObservation):

sugarPrice, sunlightIndex, importTariff, exportTariff, transportFees

Sample data provided:

prices_round_4_day_[1‑3].csv – full LOB snapshots

trades_round_4_day_[1‑3].csv – tape

observations_round_4_day_[1‑3].csv – cost factors per tick

2. Quick Data Diagnostics

Metric (MACARONS)

Day 1

Day 2

Day 3

Notes

Avg spread

7.8

8.2

8.6

Wide → profitable passive mm

Mid σ (tick‑tick)

2.54

2.59

2.59

~0.4 %

βsugar

+3.0

−2.7

+11.4

Regime flip D2

βsunlight

−2.9

+2.6

−1.0

βimportTariff

−10.5

+32.7

−36.1

Biggest elasticity

3. Baseline Provided by Sam

File baseline.py citeturn0file0

Combines (1) fundamental/MA scoring & (2) micro‑structure heuristics.

~20 tunable parameters (weights, thresholds).

Performs OK early but blew up ≈ 80 k‑tick → PnL −219 k (see screenshot).

4. Identified Pain Points

No hard risk cap → large adverse moves wipe profits.

Fixed spreads/volumes ignore volatility regime changes.

Single‑mode model can’t handle day‑to‑day β sign flips.

5. New Strategy Ideas (ready‑to‑test)

5.1 BaseMM

Rolling OLS (400 tick) to price fair‑value fv = β·X + c.

Dynamic base_spread = max(2, 0.6 σ) ; edge_clip = 3 σ.

Inventory skew: +/‑0.04 × pos.

5.2 VolKiller (Risk Layer)

Monitor equity, σ.

VaR guard: |pos|·σ ≤ 0.2 × equity else shrink quotes.

Hard stop‑loss: draw‑down > 30 k or 4 σ → market flatten.

5.3 StatArbPair

Lead‑lag (300 tick) vs RAINFOREST_RESIN.

z‑score entry ±1.5 σ, exit 0.2 σ; size 15.

5.4 Possible Extensions

Regime detector (trend vs revert) → switch maker/taker behaviour.

Multi‑product basket (add KELP).

Kalman filter instead of OLS for smoother β.

6. Reference Implementation

trader_mix.py – merges BaseMM + VolKiller + StatArbPair (max 900 ms run‑time).

All parameters grouped in top‑level PARAMS dict for quick tuning.

# local replay (provided runner)
python run_local.py --trader trader_mix.py

7. What We Need Next

Independent review of factor selection / window lengths.

Explore alternative risk metrics (e.g., EWMA σ, PnL‑Based Kelly sizing).

Consider reinforcement‑learning policy to choose between maker/taker.

Any insights from previous rounds (e.g., vouchers, flipper) that might interact with MACARONS storage/conversion.