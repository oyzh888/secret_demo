# Round 5 - Counterparty Intelligence & Multi-Strategy Trading

## Trading Products

This round introduced no new products but provided critical counterparty information:

### Existing Products
1. **Basic Commodities**
   - RAINFOREST_RESIN, KELP, CROISSANTS, JAMS
   - Position limits: 50-350
   - Known for stable prices and low volatility

2. **Composite Products**
   - PICNIC_BASKET1/2
   - Contains: CROISSANTS, JAMS, DJEMBES
   - Position limits: 60-100

3. **Volcanic Products**
   - VOLCANIC_ROCK
   - VOLCANIC_ROCK_VOUCHER_9500-10500
   - Position limits: 200-400
   - Conversion limit: 10 per day

4. **Macarons**
   - MAGNIFICENT_MACARONS
   - Position limit: 75
   - Conversion limit: 10
   - Influenced by sunlight, sugar prices, and tariffs

## Key Market Participants

### Counterparty Analysis
1. **Caesar**
   - Most active trader (29,529 trades)
   - Market maker with momentum tendencies
   - Key behavior: 
     - Buys at +0.23 ticks above mid
     - Sells at -0.14 ticks below mid
     - Shows clear trend patterns (3+ consecutive trades)

2. **Camilla**
   - Second most active (18,938 trades)
   - Strong buyer bias (13,053 buys vs 5,885 sells)
   - Potential trend follower

3. **Charlie**
   - Balanced trader (17,306 trades)
   - Neutral trading style
   - Good liquidity provider

## Strategy Overview

Our strategy focused on:
- Counterparty-based adaptive market making
- Trend detection and momentum trading
- Multi-product portfolio management
- Fee-aware position sizing

### Key Features

1. **Counterparty Intelligence**
   - Real-time tracking of major players' positions
   - Trend streak detection (3+ consecutive trades)
   - Price deviation analysis from mid-price
   - Adaptive response to counterparty behavior

2. **Product Classification**
   - Low volatility (α): RAINFOREST_RESIN, KELP, CROISSANTS, JAMS
   - Medium volatility (β): MAGNIFICENT_MACARONS
   - High volatility (γ): VOLCANIC_ROCK & Vouchers
   - Composite products (δ): PICNIC_BASKET1/2, DJEMBES

3. **Risk Management**
   - Position limits per product category
   - Fee-aware trading (0.2% → 0.5% → 1% tiers)
   - Counterparty exposure limits
   - Trend-based position sizing

## Special Notes

- This round introduced counterparty information in trades
- Focus was on understanding and exploiting counterparty behavior
- Required sophisticated pattern recognition and adaptive strategies
- Fee structure became more important with high-frequency trading
- West Archipelago trading day introduced new market dynamics