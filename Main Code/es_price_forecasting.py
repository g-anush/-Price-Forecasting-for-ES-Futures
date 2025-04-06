import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For preprocessing and feature engineering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

# For evaluation
import empyrical as ep  # For financial metrics like Sharpe ratio
import pyfolio as pf    # For portfolio analysis

# For neural networks
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Reading the data
def load_and_process_data(file_path):
    """
    Load and process the ES futures data from CSV.
    """
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Rename unnamed column to Date
    if '' in df.columns:
        df = df.rename(columns={'': 'Date'})
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%y %H:%M', errors='coerce')
    
    # Sort by timestamp
    df = df.sort_values('Timestamp')
    
    # Drop rows with missing timestamps
    df = df.dropna(subset=['Timestamp'])
    
    # Set timestamp as index
    df = df.set_index('Timestamp')
    
    print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def engineer_features(df):
    """
    Engineer features for the ES futures data.
    """
    print("Engineering features...")
    
    # Create a copy of the dataframe
    df_feat = df.copy()
    
    # Extract time-based features
    df_feat['hour'] = df_feat.index.hour
    df_feat['minute'] = df_feat.index.minute
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['day_of_month'] = df_feat.index.day
    df_feat['month'] = df_feat.index.month
    df_feat['quarter'] = df_feat.index.quarter
    
    # Calculate price based features
    df_feat['returns'] = df_feat['Close'].pct_change()
    df_feat['log_returns'] = np.log(df_feat['Close'] / df_feat['Close'].shift(1))
    
    # Calculate price ranges
    df_feat['high_low_range'] = df_feat['High'] - df_feat['Low']
    df_feat['close_open_range'] = df_feat['Close'] - df_feat['Open']
    
    # Calculate volume indicators
    df_feat['volume_change'] = df_feat['Period Volume'].pct_change()
    df_feat['volume_ma_5'] = df_feat['Period Volume'].rolling(window=5).mean()
    df_feat['volume_ma_10'] = df_feat['Period Volume'].rolling(window=10).mean()
    
    # Technical indicators
    # Moving averages
    df_feat['ma_5'] = df_feat['Close'].rolling(window=5).mean()
    df_feat['ma_10'] = df_feat['Close'].rolling(window=10).mean()
    df_feat['ma_20'] = df_feat['Close'].rolling(window=20).mean()
    df_feat['ma_50'] = df_feat['Close'].rolling(window=50).mean()
    
    # Exponential moving averages
    df_feat['ema_5'] = df_feat['Close'].ewm(span=5, adjust=False).mean()
    df_feat['ema_10'] = df_feat['Close'].ewm(span=10, adjust=False).mean()
    df_feat['ema_20'] = df_feat['Close'].ewm(span=20, adjust=False).mean()
    
    # MACD
    df_feat['ema_12'] = df_feat['Close'].ewm(span=12, adjust=False).mean()
    df_feat['ema_26'] = df_feat['Close'].ewm(span=26, adjust=False).mean()
    df_feat['macd'] = df_feat['ema_12'] - df_feat['ema_26']
    df_feat['macd_signal'] = df_feat['macd'].ewm(span=9, adjust=False).mean()
    df_feat['macd_hist'] = df_feat['macd'] - df_feat['macd_signal']
    
    # RSI (Relative Strength Index)
    delta = df_feat['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df_feat['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df_feat['bb_middle'] = df_feat['Close'].rolling(window=20).mean()
    df_feat['bb_stddev'] = df_feat['Close'].rolling(window=20).std()
    df_feat['bb_upper'] = df_feat['bb_middle'] + (df_feat['bb_stddev'] * 2)
    df_feat['bb_lower'] = df_feat['bb_middle'] - (df_feat['bb_stddev'] * 2)
    df_feat['bb_width'] = (df_feat['bb_upper'] - df_feat['bb_lower']) / df_feat['bb_middle']
    df_feat['bb_pct'] = (df_feat['Close'] - df_feat['bb_lower']) / (df_feat['bb_upper'] - df_feat['bb_lower'])
    
    # Average True Range (ATR)
    high_low = df_feat['High'] - df_feat['Low']
    high_close = np.abs(df_feat['High'] - df_feat['Close'].shift())
    low_close = np.abs(df_feat['Low'] - df_feat['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df_feat['atr_14'] = true_range.rolling(14).mean()
    
    # Momentum indicators
    df_feat['momentum_5'] = df_feat['Close'] / df_feat['Close'].shift(5) - 1
    df_feat['momentum_10'] = df_feat['Close'] / df_feat['Close'].shift(10) - 1
    
    # Lagged features
    for i in range(1, 6):
        df_feat[f'close_lag_{i}'] = df_feat['Close'].shift(i)
        df_feat[f'returns_lag_{i}'] = df_feat['returns'].shift(i)
        df_feat[f'volume_lag_{i}'] = df_feat['Period Volume'].shift(i)
    
    # Target variables for forecasting (future returns)
    for i in [1, 5, 10, 20]:
        df_feat[f'future_return_{i}'] = df_feat['Close'].pct_change(periods=i).shift(-i)
        df_feat[f'future_direction_{i}'] = np.where(df_feat[f'future_return_{i}'] > 0, 1, 0)
    
    # Drop NaN values
    df_feat = df_feat.dropna()
    
    print(f"Feature engineering completed. Dataset now has {df_feat.shape[1]} features.")
    return df_feat

def create_train_test_split(df, test_size=0.2, gap=1440):
    """
    Create train-test split for time series data with a gap.
    The gap represents the number of minutes (1440 = 1 day) to separate training and testing.
    """
    print("Creating train-test split...")
    
    # Find the split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Add gap
    if gap > 0:
        train_end_idx = split_idx - gap
    else:
        train_end_idx = split_idx
    
    # Create train and test sets
    train_df = df.iloc[:train_end_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Training set: {train_df.shape[0]} samples ({train_df.index.min()} to {train_df.index.max()})")
    print(f"Testing set: {test_df.shape[0]} samples ({test_df.index.min()} to {test_df.index.max()})")
    
    return train_df, test_df

def prepare_model_data(train_df, test_df, target_col, feature_cols=None):
    """
    Prepare data for modeling with feature scaling.
    """
    if feature_cols is None:
        # Use all columns except targets and symbols
        exclude_cols = ['Symbol', 'Request ID', 'Message ID', 'Date'] + \
                        [col for col in train_df.columns if col.startswith('future_')]
        feature_cols = [col for col in train_df.columns if col not in exclude_cols and col != target_col]
    
    # Prepare features and target
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler, feature_cols

def train_evaluate_models(X_train, y_train, X_test, y_test, is_classification=False):
    """
    Train and evaluate multiple models on the dataset.
    """
    print("Training and evaluating models...")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        metrics = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        results[name] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred_test
        }
        
        print(f"{name} - Test MSE: {metrics['test_mse']:.6f}, Test MAE: {metrics['test_mae']:.6f}, Test R²: {metrics['test_r2']:.6f}")
    
    return results

def build_lstm_model(X_train, y_train, X_test, y_test):
    """
    Build and train an LSTM model for time series forecasting.
    """
    print("Building LSTM model...")
    
    # Reshape input for LSTM [samples, time steps, features]
    X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    # Build model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(1, X_train.shape[1])))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    
    # Train model
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate model
    y_pred_train = model.predict(X_train_reshaped).flatten()
    y_pred_test = model.predict(X_test_reshaped).flatten()
    
    metrics = {
        'train_mse': mean_squared_error(y_train, y_pred_train),
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    print(f"LSTM - Test MSE: {metrics['test_mse']:.6f}, Test MAE: {metrics['test_mae']:.6f}, Test R²: {metrics['test_r2']:.6f}")
    
    return {
        'model': model,
        'metrics': metrics,
        'predictions': y_pred_test,
        'history': history
    }

def generate_trading_signals(df, model_results, target_col, scaler=None, threshold=0.0001):
    """
    Generate trading signals based on model predictions.
    """
    print("Generating trading signals...")
    
    signals_df = df.copy()
    
    # Combine all model predictions
    ensemble_preds = np.zeros(len(signals_df))
    
    for name, result in model_results.items():
        if 'predictions' in result:
            # Add model predictions to the dataframe
            signals_df[f'{name}_pred'] = np.nan
            signals_df.iloc[-len(result['predictions']):, signals_df.columns.get_loc(f'{name}_pred')] = result['predictions']
            
            # Update ensemble predictions
            valid_indices = ~np.isnan(signals_df[f'{name}_pred'])
            ensemble_preds[valid_indices] += signals_df.loc[valid_indices, f'{name}_pred'].values
    
    # Average the ensemble predictions
    if len(model_results) > 0:
        ensemble_preds = ensemble_preds / len(model_results)
        signals_df['ensemble_pred'] = ensemble_preds
    
    # Generate signals based on predictions
    for name in list(model_results.keys()) + ['ensemble']:
        pred_col = f'{name}_pred'
        if pred_col in signals_df.columns:
            # Generate position based on prediction vs threshold
            signals_df[f'{name}_position'] = np.where(signals_df[pred_col] > threshold, 1,  # Buy
                                               np.where(signals_df[pred_col] < -threshold, -1, 0))  # Sell or Hold
    
    return signals_df

def calculate_returns(signals_df, initial_capital=100000, transaction_cost=0.0001):
    """
    Calculate returns and performance metrics based on trading signals.
    """
    print("Calculating returns and performance metrics...")
    
    # Create a copy of the signals dataframe for returns calculation
    returns_df = signals_df.copy()
    
    # Calculate strategy returns for each model
    strategy_returns = {}
    
    for col in returns_df.columns:
        if col.endswith('_position'):
            model_name = col.replace('_position', '')
            
            # Calculate strategy returns (position * next period return)
            returns_df[f'{model_name}_returns'] = returns_df[col] * returns_df['future_return_1']
            
            # Apply transaction costs when position changes
            position_changes = returns_df[col].diff().abs()
            transaction_costs = position_changes * transaction_cost
            returns_df[f'{model_name}_net_returns'] = returns_df[f'{model_name}_returns'] - transaction_costs
            
            # Calculate cumulative returns
            returns_df[f'{model_name}_cum_returns'] = (1 + returns_df[f'{model_name}_net_returns']).cumprod()
            
            # Store returns series for performance calculation
            strategy_returns[model_name] = returns_df[f'{model_name}_net_returns']
    
    # Calculate performance metrics
    performance_metrics = {}
    
    for model_name, returns in strategy_returns.items():
        # Skip if no returns data
        if returns.dropna().empty:
            continue
            
        # Calculate metrics
        metrics = {
            'total_return': (returns_df[f'{model_name}_cum_returns'].iloc[-1] - 1) * 100,
            'annualized_return': ep.annual_return(returns),
            'annualized_volatility': ep.annual_volatility(returns),
            'sharpe_ratio': ep.sharpe_ratio(returns),
            'sortino_ratio': ep.sortino_ratio(returns),
            'max_drawdown': ep.max_drawdown(returns),
            'calmar_ratio': ep.calmar_ratio(returns),
            'omega_ratio': ep.omega_ratio(returns)
        }
        
        performance_metrics[model_name] = metrics
    
    # Calculate buy-and-hold returns for comparison
    returns_df['buy_hold_returns'] = returns_df['future_return_1']
    returns_df['buy_hold_cum_returns'] = (1 + returns_df['buy_hold_returns']).cumprod()
    
    buy_hold_returns = returns_df['buy_hold_returns']
    performance_metrics['buy_hold'] = {
        'total_return': (returns_df['buy_hold_cum_returns'].iloc[-1] - 1) * 100,
        'annualized_return': ep.annual_return(buy_hold_returns),
        'annualized_volatility': ep.annual_volatility(buy_hold_returns),
        'sharpe_ratio': ep.sharpe_ratio(buy_hold_returns),
        'sortino_ratio': ep.sortino_ratio(buy_hold_returns),
        'max_drawdown': ep.max_drawdown(buy_hold_returns),
        'calmar_ratio': ep.calmar_ratio(buy_hold_returns),
        'omega_ratio': ep.omega_ratio(buy_hold_returns)
    }
    
    return returns_df, performance_metrics

def visualize_results(returns_df, performance_metrics):
    """
    Visualize the trading results and performance metrics.
    """
    print("Visualizing results...")
    
    # Set up plots
    plt.figure(figsize=(14, 18))
    
    # Plot 1: Cumulative returns
    plt.subplot(3, 1, 1)
    for col in returns_df.columns:
        if col.endswith('_cum_returns'):
            model_name = col.replace('_cum_returns', '')
            plt.plot(returns_df.index, returns_df[col], label=f'{model_name}')
    
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Drawdowns
    plt.subplot(3, 1, 2)
    for model_name in performance_metrics.keys():
        if f'{model_name}_net_returns' in returns_df.columns:
            drawdown = ep.roll_max_drawdown(returns_df[f'{model_name}_net_returns'], window=252)
            plt.plot(returns_df.index, drawdown, label=f'{model_name}')
    
    plt.title('Rolling Drawdowns (252-day window)')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Performance metrics comparison
    plt.subplot(3, 1, 3)
    metrics_to_plot = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
    
    metric_data = {}
    for metric in metrics_to_plot:
        metric_data[metric] = [performance_metrics[model][metric] for model in performance_metrics]
    
    bar_width = 0.25
    r = np.arange(len(performance_metrics))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(r + i*bar_width, metric_data[metric], width=bar_width, label=metric.replace('_', ' ').title())
    
    plt.title('Performance Metrics Comparison')
    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.xticks(r + bar_width, performance_metrics.keys(), rotation=45)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_visualization.png')
    plt.close()
    
    # Create a performance summary table
    performance_summary = pd.DataFrame(performance_metrics).T
    
    return performance_summary

def feature_importance(model_results, feature_cols):
    """
    Extract and visualize feature importance from tree-based models.
    """
    importance_df = pd.DataFrame()
    
    for name, result in model_results.items():
        if hasattr(result['model'], 'feature_importances_'):
            importances = result['model'].feature_importances_
            importance_dict = dict(zip(feature_cols, importances))
            importance_df[name] = pd.Series(importance_dict)
    
    if not importance_df.empty:
        # Sort by average importance
        importance_df['Average'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('Average', ascending=False)
        
        # Plot the top features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15).index
        
        for name in importance_df.columns:
            if name != 'Average':
                plt.plot(top_features, importance_df.loc[top_features, name], 'o-', label=name)
        
        plt.title('Feature Importance Across Models')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Importance')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    return importance_df

def main():
    # File path
    file_path = 'cme.es.minute_bars_data.2020.csv'
    
    # Load and process data
    df = load_and_process_data(file_path)
    
    # Engineer features
    df_features = engineer_features(df)
    
    # Create train-test split (80% train, 20% test with 1-day gap)
    train_df, test_df = create_train_test_split(df_features, test_size=0.2, gap=1440)
    
    # Target for prediction: 10-minute future return
    target_col = 'future_return_10'
    
    # Prepare data for modeling
    X_train, y_train, X_test, y_test, scaler, feature_cols = prepare_model_data(train_df, test_df, target_col)
    
    # Train and evaluate traditional models
    model_results = train_evaluate_models(X_train, y_train, X_test, y_test)
    
    # Train LSTM model
    lstm_result = build_lstm_model(X_train, y_train, X_test, y_test)
    model_results['LSTM'] = lstm_result
    
    # Generate trading signals
    signals_df = generate_trading_signals(df_features, model_results, target_col)
    
    # Calculate returns and performance metrics
    returns_df, performance_metrics = calculate_returns(signals_df)
    
    # Visualize results
    performance_summary = visualize_results(returns_df, performance_metrics)
    
    # Extract feature importance
    importance_df = feature_importance(model_results, feature_cols)
    
    # Save results
    performance_summary.to_csv('performance_summary.csv')
    importance_df.to_csv('feature_importance.csv')
    
    # Print final summary
    print("\n=== MODEL PERFORMANCE SUMMARY ===")
    print(performance_summary)
    
    return performance_summary, model_results, returns_df

if __name__ == "__main__":
    main()