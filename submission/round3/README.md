# Round 3 - Volcanic Rock Vouchers

## Trading Products

This round introduced volcanic rock vouchers and the underlying volcanic rock:

### Volcanic Rock
- **VOLCANIC_ROCK**
  - Position limit: 400
  - The underlying asset for all vouchers

### Volcanic Rock Vouchers
Five different vouchers with varying strike prices:
1. **VOLCANIC_ROCK_VOUCHER_9500**
   - Position limit: 200
   - Strike price: 9,500 SeaShells
   - 7 days to expiry (starting from round 1)

2. **VOLCANIC_ROCK_VOUCHER_9750**
   - Position limit: 200
   - Strike price: 9,750 SeaShells
   - 7 days to expiry

3. **VOLCANIC_ROCK_VOUCHER_10000**
   - Position limit: 200
   - Strike price: 10,000 SeaShells
   - 7 days to expiry

4. **VOLCANIC_ROCK_VOUCHER_10250**
   - Position limit: 200
   - Strike price: 10,250 SeaShells
   - 7 days to expiry

5. **VOLCANIC_ROCK_VOUCHER_10500**
   - Position limit: 200
   - Strike price: 10,500 SeaShells
   - 7 days to expiry

## Strategy Overview

Our strategy focused on:
- Options pricing and volatility trading
- Implied volatility surface analysis
- Time decay and strike price relationships

Key features:
- Implemented Black-Scholes implied volatility calculations
- Monitored volatility surface patterns
- Tracked base IV (implied volatility at m_t = 0)
- Executed volatility arbitrage between different strikes

## Special Notes

- This round introduced options-like instruments
- Focus was on understanding and exploiting volatility patterns
- Required sophisticated pricing models and risk management
- Vouchers had 7 days to expiry at the start, reducing to 2 days by round 5 