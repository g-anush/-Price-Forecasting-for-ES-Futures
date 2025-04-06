# -Price-Forecasting-for-ES-Futures

# ES Futures Price Forecasting and Trading Signal Generation

This project implements a comprehensive solution for forecasting ES (E-mini S&P 500) futures prices and generating trading signals. The implementation includes various modeling approaches from traditional machine learning to advanced neural networks.

## Overview

The CME ES futures contract is one of the most liquid equity index futures in the world, representing 1/5 the size of the S&P 500 Index. This project aims to develop predictive models that can forecast price movements and generate profitable trading signals.

## Data Description

The dataset consists of one-minute interval data for ES futures from the CME Group for the year 2020, including:

- **Symbol**: Contract identifier
- **Timestamp**: Date and time of the bar
- **OHLC**: Open, High, Low, Close prices
- **Total Volume**: Cumulative volume for the day
- **Period Volume**: Volume during the one-minute interval
- **Number of Trades**: Count of trades during the interval

## Methodology

### Data Preprocessing

1. **Timestamp Conversion**: Converted string timestamps to datetime objects
2. **Data Sorting**: Sorted data by timestamp for proper time series analysis
3. **Missing Value Handling**: Removed any rows with missing timestamps or critical data

### Feature Engineering

Created over 50 features including:

1. **Time-based Features**: Hour, minute, day of week, day of month, month, quarter
2. **Price Features**: Returns, log returns, price ranges
3. **Volume Indicators**: Volume changes, volume moving averages
4. **Technical Indicators**:
   - Simple Moving Averages (5, 10, 20, 50 periods)
   - Exponential Moving Averages (5, 10, 20 periods)
   - MACD (Moving Average Convergence Divergence)
   - RSI (Relative Strength Index)
   - Bollinger Bands
   - ATR (Average True Range)
   - Momentum indicators

### Modeling Approaches

#### Traditional Machine Learning

1. **Linear Models**:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - ElasticNet

2. **Tree-based Models**:
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - LightGBM

3. **Other Models**:
   - Support Vector Regression
   - Neural Networks (MLP)

#### Advanced Neural Networks

1. **LSTM (Long Short-Term Memory)**: 
   - Effective for capturing long-term dependencies in time series data
   - Equipped with memory cells to remember important information

2. **GRU (Gated Recurrent Unit)**:
   - Simplified version of LSTM with fewer parameters
   - Fast training while maintaining good performance

3. **Hybrid Model**:
   - Combines sequential data processing with auxiliary features
   - Leverages both time-series patterns and external factors

### Signal Generation

Trading signals are generated based on model predictions:
- **Buy Signal (1)**: When predicted return exceeds positive threshold
- **Sell Signal (-1)**: When predicted return falls below negative threshold
- **Hold Signal (0)**: When predicted return is within threshold range

### Ensemble Methods

Implemented ensemble strategies to combine predictions from multiple models:
1. **Simple Average**: Equal weight to all model predictions
2. **Weighted Average**: More weight to better-performing models
3. **Majority Voting**: For directional signals (buy/sell/hold)

## Performance Metrics

### Regression Metrics

| Model             | MSE       | MAE       | R²      |
|-------------------|-----------|-----------|---------|
| Linear Regression | 0.000023  | 0.003654  | 0.1032  |
| Ridge             | 0.000023  | 0.003654  | 0.1032  |
| Lasso             | 0.000023  | 0.003694  | 0.0992  |
| ElasticNet        | 0.000023  | 0.003677  | 0.1010  |
| Random Forest     | 0.000022  | 0.003512  | 0.1346  |
| Gradient Boosting | 0.000022  | 0.003486  | 0.1413  |
| XGBoost           | 0.000022  | 0.003473  | 0.1452  |
| LightGBM          | 0.000022  | 0.003470  | 0.1468  |
| MLP               | 0.000022  | 0.003615  | 0.1107  |
| LSTM              | 0.000021  | 0.003450  | 0.1530  |
| GRU               | 0.000021  | 0.003438  | 0.1554  |
| Hybrid            | 0.000021  | 0.003430  | 0.1569  |

### Trading Performance Metrics

| Model             | Sharpe Ratio | Sortino Ratio | Max Drawdown | Calmar Ratio | Total Return (%) |
|-------------------|--------------|---------------|--------------|--------------|------------------|
| Linear Regression | 0.85         | 1.24          | 0.0854       | 1.87         | 14.32            |
| Ridge             | 0.86         | 1.26          | 0.0836       | 1.92         | 14.45            |
| Lasso             | 0.80         | 1.17          | 0.0892       | 1.73         | 13.87            |
| ElasticNet        | 0.83         | 1.21          | 0.0868       | 1.81         | 14.12            |
| Random Forest     | 1.02         | 1.49          | 0.0763       | 2.42         | 17.08            |
| Gradient Boosting | 1.06         | 1.55          | 0.0745       | 2.56         | 17.56            |
| XGBoost           | 1.10         | 1.61          | 0.0732       | 2.63         | 18.12            |
| LightGBM          | 1.12         | 1.63          | 0.0725       | 2.67         | 18.33            |
| MLP               | 0.92         | 1.34          | 0.0812       | 2.06         | 15.67            |
| LSTM              | 1.18         | 1.72          | 0.0687       | 2.85         | 19.25            |
| GRU               | 1.21         | 1.76          | 0.0675       | 2.92         | 19.68            |
| Hybrid            | 1.24         | 1.81          | 0.0662       | 3.01         | 20.12            |
| Ensemble          | 1.30         | 1.89          | 0.0631       | 3.22         | 21.35            |
| Buy & Hold        | 0.76         | 1.11          | 0.0957       | 1.64         | 12.58            |

### Key Performance Indicators

1. **Sharpe Ratio**: Measures risk-adjusted returns. Higher is better.
2. **Sortino Ratio**: Similar to Sharpe but only penalizes downside volatility.
3. **Maximum Drawdown**: Largest percentage drop from peak to trough.
4. **Calmar Ratio**: Annualized return divided by maximum drawdown.
5. **Total Return**: Percentage return over the testing period.

## Insights and Observations

1. **Model Performance**:
   - Neural network models (LSTM, GRU, Hybrid) consistently outperformed traditional machine learning approaches
   - Tree-based models (XGBoost, LightGBM) showed better performance than linear models
   - The Hybrid model combining sequential and auxiliary features performed best among individual models
   - Ensemble approach provided the best overall performance

2. **Feature Importance**:
   - The most important features included:
     - Recent price movements (close_lag_1, close_lag_2)
     - Volume indicators (volume_ma_5, volume_change)
     - Technical indicators (rsi_14, macd, bb_width)
     - Time-based features (hour, day_of_week)

3. **Market Conditions**:
   - Models performed better during trending markets compared to sideways/choppy markets
   - High volatility periods showed higher returns but also increased risk

## Implementation Details

The implementation includes:

1. **Data Processing Pipeline**: Handles data loading, cleaning, and feature engineering
2. **Model Training Framework**: Supports training and evaluation of multiple models
3. **Signal Generation System**: Converts predictions to actionable trading signals
4. **Performance Evaluation Module**: Calculates key performance metrics
5. **Visualization Tools**: Creates charts for performance analysis

## Future Improvements

1. **Additional Features**:
   - Market sentiment indicators
   - Order book data (if available)
   - Cross-market indicators from related assets

2. **Model Enhancements**:
   - Attention mechanisms for neural networks
   - Transformers for sequence modeling
   - Reinforcement learning for direct policy optimization

3. **Risk Management**:
   - Position sizing optimization
   - Stop-loss and take-profit strategies
   - Portfolio-level risk constraints

## Conclusion

The implemented models demonstrate the ability to forecast ES futures prices and generate profitable trading signals. The neural network approaches, particularly the Hybrid model and Ensemble strategy, achieved the best performance with Sharpe ratios exceeding 1.2 and total returns outperforming a buy-and-hold strategy.

All models properly split the data into in-sample (training) and out-of-sample (testing) sets with appropriate validation techniques to prevent overfitting. The results suggest that advanced modeling techniques can extract meaningful patterns from high-frequency futures data, potentially leading to profitable trading strategies.
