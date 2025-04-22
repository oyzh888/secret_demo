# Round 2 - Picnic Baskets and Components

## Trading Products

This round introduced picnic baskets and their individual components:

### Picnic Baskets
1. **PICNIC_BASKET1**
   - Contains: 6 CROISSANTS, 3 JAMS, 1 DJEMBE
   - Position limit: 60

2. **PICNIC_BASKET2**
   - Contains: 4 CROISSANTS, 2 JAMS
   - Position limit: 100

### Individual Products
1. **CROISSANTS**
   - Position limit: 250
   - Component of both picnic baskets

2. **JAMS**
   - Position limit: 350
   - Component of both picnic baskets

3. **DJEMBES**
   - Position limit: 60
   - Only in PICNIC_BASKET1

## Strategy Overview

Our strategy focused on:
- Arbitrage between basket prices and their components
- Market making for individual products
- Basket assembly/disassembly based on price differentials

Key features:
- Implemented basket valuation based on component prices
- Monitored price relationships between baskets and components
- Executed basket assembly when profitable
- Maintained balanced positions across all products

## Special Notes

- This round introduced the concept of composite products
- Focus was on identifying and exploiting price inefficiencies between baskets and components
- Required careful position management to maintain basket assembly/disassembly flexibility 