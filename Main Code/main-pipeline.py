#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ES Futures Price Forecasting and Trading Signal Pipeline

This script implements a complete pipeline for forecasting ES futures prices
and generating trading signals using multiple modeling approaches.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import core modules
from es_price_forecasting import (
    load_and_process_data, 
    engineer_features, 
    create_train_test_split,
    prepare_model_data,
    train_evaluate_models,
    build_lstm_model,
    generate_trading_signals,
    calculate_returns,
    visualize_results,
    feature_importance
)

# Import neural network modules
from neural_network_model import (
    DeepLearningModels,
    select_features,
    split_features,
    train_neural_networks,
    generate_nn_trading_signals
)

def main():
    """
    Main function to run the complete pipeline.
    """
    print("="*80)
    print(" ES FUTURES PRICE FORECASTING AND TRADING SIGNAL PIPELINE ")
    print("="*80)
    
    # File path
    file_path = 'cme.es.minute_bars_data.2020.csv'
    
    # Step 1: Load and process data
    print("\nSTEP 1: Loading and processing data...")
    df = load_and_process_data(file_path)
    print(f"Dataset shape after loading: {df.shape}")
    
    # Step 2: Engineer features
    print("\nSTEP 2: Engineering features...")
    df_features = engineer_features(df)
    print(f"Dataset shape after feature engineering: {df_features.shape}")
    
    # Check for any missing values
    missing_values = df_features.isnull().sum()
    print(f"Columns with missing values: {missing_values[missing_values > 0]}")
    
    # Fill or drop missing values if any
    if missing_values.sum() > 0:
        print("Dropping rows with missing values...")
        df_features = df_features.dropna()
        print(f"Dataset shape after dropping missing values: {df_features.shape}")
    
    # Step 3: Create train-test split
    print("\nSTEP 3: Creating train-test split...")
    train_df, test_df = create_train_test_split(df_features, test_size=0.2, gap=1440)
    
    # Step 4: Train traditional models
    print("\nSTEP 4: Training traditional machine learning models...")
    
    # Define target columns for prediction
    target_cols = ['future_return_1', 'future_return_5', 'future_return_10', 'future_return_20']
    
    # Dictionary to store results for each target
    all_model_results = {}
    all_performance_metrics = {}
    
    for target_col in target_cols:
        print(f"\nTraining models for target: {target_col}")
        
        # Prepare data for modeling
        X_train, y_train, X_test, y_test, scaler, feature_cols = prepare_model_data(
            train_df, test_df, target_col
        )
        
        # Train and evaluate traditional models
        model_results = train_evaluate_models(X_train, y_train, X_test, y_test)
        
        # Build LSTM model
        lstm_result = build_lstm_model(X_train, y_train, X_test, y_test)
        model_results['LSTM'] = lstm_result
        
        # Store results
        all_model_results[target_col] = model_results
    
    # Step 5: Train neural network models
    print("\nSTEP 5: Training advanced neural network models...")
    
    # Use 10-minute future return as the default target
    default_target = 'future_return_10'
    
    # Train neural network models
    dl_models, nn_results = train_neural_networks(
        train_df, target_col=default_target, test_size=0.2, lookback=20
    )
    
    # Step 6: Generate trading signals and evaluate performance
    print("\nSTEP 6: Generating trading signals and evaluating performance...")
    
    # Generate signals using traditional models
    signals_df = generate_trading_signals(
        df_features, all_model_results[default_target], default_target
    )
    
    # Generate signals using neural network models
    nn_signals_df = generate_nn_trading_signals(test_df, nn_results, lookback=20)
    
    # Combine signals
    for col in nn_signals_df.columns:
        if col.endswith('_pred') or col.endswith('_position'):
            signals_df[col] = nn_signals_df[col]
    
    # Calculate returns and performance metrics
    returns_df, performance_metrics = calculate_returns(signals_df)
    
    # Step 7: Visualize results
    print("\nSTEP 7: Visualizing results...")
    performance_summary = visualize_results(returns_df, performance_metrics)
    
    # Step 8: Feature importance analysis
    print("\nSTEP 8: Analyzing feature importance...")
    importance_df = feature_importance(
        all_model_results[default_target], 
        feature_cols
    )
    
    # Step 9: Save results
    print("\nSTEP 9: Saving results...")
    os.makedirs('results', exist_ok=True)
    
    # Save performance metrics
    performance_summary.to_csv('results/performance_summary.csv')
    
    # Save feature importance
    if importance_df is not None:
        importance_df.to_csv('results/feature_importance.csv')
    
    # Save predictions
    predictions_df = pd.DataFrame(index=test_df.index)
    
    for model_name, result in all_model_results[default_target].items():
        if 'predictions' in result:
            predictions_df[f'{model_name}_pred'] = np.nan
            predictions_df.iloc[-len(result['predictions']):, predictions_df.columns.get_loc(f'{model_name}_pred')] = result['predictions']
    
    for model_name, result in nn_results.items():
        predictions_df[f'NN_{model_name}_pred'] = np.nan
        predictions_df.iloc[-len(result['predictions']):, predictions_df.columns.get_loc(f'NN_{model_name}_pred')] = result['predictions']
    
    predictions_df.to_csv('results/model_predictions.csv')
    
    # Save trading signals and positions
    signals_cols = [col for col in signals_df.columns if col.endswith('_position') or col.endswith('_pred')]
    signals_df[signals_cols].to_csv('results/trading_signals.csv')
    
    # Save returns
    returns_cols = [col for col in returns_df.columns if col.endswith('_returns')]
    returns_df[returns_cols].to_csv('results/strategy_returns.csv')
    
    print("\nPipeline execution completed successfully!")
    print("\nSummary of performance metrics:")
    print(performance_summary[['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'total_return']].round(3))
    
    return performance_summary, all_model_results, returns_df

if __name__ == "__main__":
    main()
