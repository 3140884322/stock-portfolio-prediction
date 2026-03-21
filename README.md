# Stock Prediction and Portfolio Construction

## Overview
This project builds on the StockMixer framework to predict stock returns and extends it to portfolio-level decision making.

Instead of evaluating predictions only at the individual stock level, we construct a portfolio by selecting top-ranked stocks and measure its performance using cumulative returns and Sharpe ratio.

## Methodology
1. Use StockMixer to predict stock-level returns
2. Rank stocks based on predicted return
3. Select top-K stocks (K=5)
4. Construct equal-weight portfolio
5. Evaluate performance:
   - Mean daily return
   - Cumulative return
   - Sharpe ratio

## Files
- `train.py`: main training script with portfolio logic
- `model.py`: StockMixer model definition
- `evaluator.py`: evaluation metrics
- `load_data.py`: data loading functions

## Dataset
Dataset is not included due to size. Please use the original StockMixer dataset and place it in the `/dataset` directory.

## Contribution
This project extends the original StockMixer model by adding a portfolio construction and evaluation module based on predicted returns.
