# IMC Prosperity Round 4 Strategy

## MAGNIFICENT_MACARONS Trading Algorithm

This is our trading strategy for MAGNIFICENT_MACARONS in IMC Prosperity Round 4. The strategy combines market making with fundamental analysis to adaptively trade based on current market conditions.

### Key Features

1. **Adaptive Market Making**
   - Dynamically adjusts bid-ask spreads based on price volatility
   - Asymmetric spreads based on current inventory position
   - Scales order sizes to manage inventory risk

2. **Market Regime Detection**
   - Identifies different market regimes (TARIFF_UP, TARIFF_DOWN, COST_UP, COST_DOWN, UPTREND, DOWNTREND, RANGE)
   - Uses price trends and fundamental indicators to detect regime shifts
   - Adapts strategy parameters based on current regime

3. **Fundamental Analysis**
   - Incorporates sugar price, sunlight index, and tariff data
   - Weights fundamental factors differently based on detected regime
   - Adjusts fair value estimation in response to changing fundamentals

4. **Position Management**
   - Manages inventory within position limit (±75 lot)
   - Uses conversions strategically (limit: 10 lot per iteration)
   - Reduces order sizes as position approaches limits

### Strategy Logic

The strategy works as follows:

1. Maintains a rolling window of price history and calculates volatility
2. Computes short and long exponential moving averages (EMAs) to identify trends
3. Detects market regime by analyzing price trends and fundamental changes
4. Calculates fair value based on market data and fundamentals with regime-specific adjustments
5. Sets bid/ask prices with optimal spreads based on volatility and inventory
6. Executes against favorable market orders and places limit orders for market making
7. Manages physical conversion when appropriate

### Performance Considerations

This strategy is designed to:
- Profit from wide spreads in the MAGNIFICENT_MACARONS market
- Adjust quickly to tariff changes which cause significant price movements
- Balance inventory risk with profit opportunities
- Adapt to changing market correlations with fundamental factors

The strategy relies on Python's NumPy and Pandas libraries for efficient calculations while staying within the 900ms runtime limit.

# Round 4 - Magnificent Macarons

## Trading Products

This round introduced a luxury product with complex pricing factors:

### Magnificent Macarons
- **MAGNIFICENT_MACARONS**
  - Position limit: 75
  - Conversion limit: 10
  - Delicacy produced on the island of Pristine Cuisine

## Price Influencing Factors

The value of Magnificent Macarons is influenced by multiple factors:
1. **Sunlight Index**
   - Critical impact on production
   - Existence of Critical Sunlight Index (CSI)
   - Below CSI: Prices increase substantially
   - Above CSI: Prices trade around fair value

2. **Sugar Prices**
   - Main ingredient
   - Price correlation with sunlight index
   - Impact on production costs

3. **Logistics Costs**
   - Shipping costs
   - Import/export tariffs
   - Storage costs (0.1 seashell per macaron)

## Strategy Overview

Our strategy focused on:
- Monitoring and analyzing sunlight index patterns
- Identifying the Critical Sunlight Index (CSI)
- Tracking sugar price correlations
- Managing storage and logistics costs

Key features:
- Implemented sunlight index monitoring system
- Developed CSI detection algorithm
- Price prediction based on environmental factors
- Risk management for storage costs

## Special Notes

- This round introduced environmental factor-based trading
- Focus was on understanding the relationship between sunlight and prices
- Required careful monitoring of multiple market factors
- Storage costs needed to be factored into trading decisions 