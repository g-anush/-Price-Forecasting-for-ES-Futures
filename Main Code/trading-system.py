#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ES Futures Trading System

This script implements a trading system that uses the trained models to
generate trading signals for ES futures in a production environment.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
import time
import logging
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('es_trading_system')

class ESFuturesTradingSystem:
    """
    A trading system for ES futures that uses trained models to generate signals.
    """
    
    def __init__(self, config_path='trading-config.json'):
        """
        Initialize the trading system.
        
        Parameters:
        -----------
        config_path : str
            Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.data_buffer = pd.DataFrame()
        self.current_position = 0
        self.trades = []
        self.last_signal_time = None
        
        # Load models and scalers
        self._load_models()
        
        logger.info(f"Trading system initialized with {len(self.models)} models")
    
    def _load_config(self, config_path):
        """
        Load configuration from a JSON file.
        
        Parameters:
        -----------
        config_path : str
            Path to the configuration file.
            
        Returns:
        --------
        config : dict
            Configuration dictionary.
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Default configuration
            return {
                "models_dir": "models",
                "data_dir": "data",
                "buffer_size": 100,
                "signal_threshold": 0.0001,
                "min_signal_interval_minutes": 15,
                "position_size": 1,
                "max_position": 3,
                "stop_loss_pct": 0.002,
                "take_profit_pct": 0.003,
                "trading_hours": {
                    "start": "09:30",
                    "end": "16:00"
                },
                "feature_engineering": {
                    "lookback_periods": [5, 10, 20, 50],
                    "volume_periods": [5, 10],
                    "rsi_period": 14,
                    "macd_params": {
                        "fast": 12,
                        "slow": 26,
                        "signal": 9
                    }
                },
                "models_to_use": ["ensemble", "XGBoost", "LSTM", "GRU"]
            }
    
    def _load_models(self):
        """
        Load trained models and scalers from files.
        """
        models_dir = self.config["models_dir"]
        
        if not os.path.exists(models_dir):
            logger.error(f"Models directory {models_dir} does not exist")
            return
        
        # Load feature columns
        feature_columns_path = os.path.join(models_dir, "feature_columns.pkl")
        if os.path.exists(feature_columns_path):
            with open(feature_columns_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
            logger.info(f"Loaded {len(self.feature_columns)} feature columns")
        
        # Load models and scalers
        for model_name in self.config["models_to_use"]:
            model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
            scaler_path = os.path.join(models_dir, f"{model_name}_scaler.pkl")
            
            # Load model
            if os.path.exists(model_path):
                try:
                    if model_name in ["LSTM", "GRU"]:
                        self.models[model_name] = tf.keras.models.load_model(model_path)
                    else:
                        with open(model_path, 'rb') as f:
                            self.models[model_name] = pickle.load(f)
                    
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {e}")
            
            # Load scaler
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, 'rb') as f:
                        self.scalers[model_name] = pickle.load(f)
                    
                    logger.info(f"Loaded scaler: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading scaler {model_name}: {e}")
    
    def _engineer_features(self, data):
        """
        Engineer features from raw data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw data with OHLCV information.
            
        Returns:
        --------
        features : pandas.DataFrame
            Engineered features.
        """
        df = data.copy()
        
        # Set datetime index if not already
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df.set_index('Timestamp', inplace=True)
        
        # Extract time-based features
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Calculate price ranges
        df['high_low_range'] = df['High'] - df['Low']
        df['close_open_range'] = df['Close'] - df['Open']
        
        # Calculate volume indicators
        if 'Volume' in df.columns:
            volume_col = 'Volume'
        elif 'Period Volume' in df.columns:
            volume_col = 'Period Volume'
        else:
            volume_col = None
        
        if volume_col:
            df['volume_change'] = df[volume_col].pct_change()
            
            for period in self.config["feature_engineering"]["volume_periods"]:
                df[f'volume_ma_{period}'] = df[volume_col].rolling(window=period).mean()
        
        # Moving averages
        for period in self.config["feature_engineering"]["lookback_periods"]:
            df[f'ma_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # MACD
        macd_params = self.config["feature_engineering"]["macd_params"]
        df['ema_fast'] = df['Close'].ewm(span=macd_params["fast"], adjust=False).mean()
        df['ema_slow'] = df['Close'].ewm(span=macd_params["slow"], adjust=False).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=macd_params["signal"], adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        rsi_period = self.config["feature_engineering"]["rsi_period"]
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        df['bb_stddev'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_stddev'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_stddev'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pct'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Lagged features
        for i in range(1, 6):
            df[f'close_lag_{i}'] = df['Close'].shift(i)
            df[f'returns_lag_{i}'] = df['returns'].shift(i)
            if volume_col:
                df[f'volume_lag_{i}'] = df[volume_col].shift(i)
        
        # Drop NaN values
        df = df.dropna()
        
        # Select only the feature columns used by the models
        if self.feature_columns:
            available_cols = set(df.columns) & set(self.feature_columns)
            missing_cols = set(self.feature_columns) - set(df.columns)
            
            if missing_cols:
                logger.warning(f"Missing feature columns: {missing_cols}")
            
            df = df[list(available_cols)]
        
        return df
    
    def _prepare_model_input(self, features, model_name):
        """
        Prepare input data for a specific model.
        
        Parameters:
        -----------
        features : pandas.DataFrame
            Engineered features.
        model_name : str
            Name of the model.
            
        Returns:
        --------
        model_input : numpy.ndarray
            Prepared input for the model.
        """
        if model_name not in self.models or model_name not in self.scalers:
            logger.error(f"Model or scaler {model_name} not found")
            return None
        
        # Apply scaler
        X = features.values
        X_scaled = self.scalers[model_name].transform(X)
        
        # Reshape for sequence models
        if model_name in ["LSTM", "GRU"]:
            # Determine lookback period, default to 20
            lookback = 20
            
            # Only prepare sequence if we have enough data
            if len(X_scaled) >= lookback:
                # Create sequence of last lookback steps
                X_seq = X_scaled[-lookback:].reshape(1, lookback, X_scaled.shape[1])
                return X_seq
            else:
                logger.warning(f"Not enough data for sequence model {model_name}")
                return None
        
        # For non-sequence models, just return the last row
        return X_scaled[-1:].reshape(1, -1)
    
    def _generate_signal(self, features):
        """
        Generate trading signal based on model predictions.
        
        Parameters:
        -----------
        features : pandas.DataFrame
            Engineered features.
            
        Returns:
        --------
        signal : int
            Trading signal (1: Buy, -1: Sell, 0: Hold).
        confidence : float
            Confidence level of the signal (0-1).
        """
        if not self.models:
            logger.error("No models available")
            return 0, 0.0
        
        # Get predictions from each model
        predictions = {}
        
        for model_name, model in self.models.items():
            if model_name == "ensemble":
                continue  # Skip ensemble as it's not a real model
            
            # Prepare input
            model_input = self._prepare_model_input(features, model_name)
            
            if model_input is not None:
                try:
                    # Get prediction
                    pred = model.predict(model_input)
                    
                    # Store prediction
                    if isinstance(pred, np.ndarray):
                        predictions[model_name] = pred[0][0] if pred.ndim > 1 else pred[0]
                    else:
                        predictions[model_name] = pred
                        
                    logger.debug(f"{model_name} prediction: {predictions[model_name]}")
                except Exception as e:
                    logger.error(f"Error getting prediction from {model_name}: {e}")
        
        # Calculate ensemble prediction if enabled
        if "ensemble" in self.config["models_to_use"] and predictions:
            predictions["ensemble"] = sum(predictions.values()) / len(predictions)
        
        # Determine final signal
        threshold = self.config["signal_threshold"]
        
        # Use model with highest weight (for now, simple average)
        final_prediction = sum(predictions.values()) / len(predictions) if predictions else 0
        
        if final_prediction > threshold:
            signal = 1  # Buy
        elif final_prediction < -threshold:
            signal = -1  # Sell
        else:
            signal = 0  # Hold
        
        # Calculate confidence based on how far the prediction is from 0
        confidence = min(abs(final_prediction) * 10, 1.0)
        
        logger.info(f"Generated signal: {signal} with confidence: {confidence:.2f}")
        
        return signal, confidence
    
    def _should_generate_signal(self, current_time):
        """
        Check if a new signal should be generated based on time constraints.
        
        Parameters:
        -----------
        current_time : datetime
            Current time.
            
        Returns:
        --------
        should_generate : bool
            Whether a new signal should be generated.
        """
        # Check if within trading hours
        trading_hours = self.config["trading_hours"]
        start_time = datetime.strptime(trading_hours["start"], "%H:%M").time()
        end_time = datetime.strptime(trading_hours["end"], "%H:%M").time()
        
        if not (start_time <= current_time.time() <= end_time):
            return False
        
        # Check if enough time has passed since last signal
        min_interval = self.config["min_signal_interval_minutes"]
        
        if self.last_signal_time is not None:
            time_diff = (current_time - self.last_signal_time).total_seconds() / 60
            if time_diff < min_interval:
                return False
        
        return True
    
    def _execute_trade(self, signal, price, timestamp, confidence):
        """
        Execute a trade based on the signal.
        
        Parameters:
        -----------
        signal : int
            Trading signal (1: Buy, -1: Sell, 0: Hold).
        price : float
            Current price.
        timestamp : datetime
            Current timestamp.
        confidence : float
            Confidence level of the signal.
            
        Returns:
        --------
        success : bool
            Whether the trade was executed successfully.
        """
        # Check if we should take action
        if signal == 0:
            return False
        
        # Calculate position size
        position_size = self.config["position_size"]
        
        # Scale position size by confidence if desired
        if self.config.get("scale_by_confidence", False):
            position_size = int(position_size * confidence)
            position_size = max(position_size, 1)  # Ensure at least 1 contract
        
        # Check max position limits
        max_position = self.config["max_position"]
        
        if signal > 0:  # Buy
            if self.current_position + position_size > max_position:
                position_size = max(0, max_position - self.current_position)
        elif signal < 0:  # Sell
            if self.current_position - position_size < -max_position:
                position_size = max(0, self.current_position + max_position)
        
        # Execute trade
        if position_size > 0:
            # Record trade
            trade = {
                "timestamp": timestamp,
                "action": "BUY" if signal > 0 else "SELL",
                "price": price,
                "size": position_size,
                "confidence": confidence,
                "position_before": self.current_position,
                "position_after": self.current_position + (position_size * signal)
            }
            
            self.trades.append(trade)
            
            # Update position
            self.current_position += (position_size * signal)
            
            # Update last signal time
            self.last_signal_time = timestamp
            
            logger.info(f"Executed trade: {trade['action']} {position_size} contracts at {price}")
            
            return True
        
        return False
    
    def _check_stop_loss_take_profit(self, current_price):
        """
        Check if stop loss or take profit should be triggered.
        
        Parameters:
        -----------
        current_price : float
            Current price.
            
        Returns:
        --------
        action : str
            Action to take ('BUY', 'SELL', or None).
        size : int
            Number of contracts to trade.
        """
        if not self.trades or self.current_position == 0:
            return None, 0
        
        # Get average entry price for current position
        long_trades = [t for t in self.trades if t["action"] == "BUY" and t["position_after"] > t["position_before"]]
        short_trades = [t for t in self.trades if t["action"] == "SELL" and t["position_after"] < t["position_before"]]
        
        if self.current_position > 0 and long_trades:
            # We have a long position
            avg_entry = sum(t["price"] * (t["position_after"] - t["position_before"]) for t in long_trades) / self.current_position
            
            # Check stop loss
            stop_loss = avg_entry * (1 - self.config["stop_loss_pct"])
            if current_price <= stop_loss:
                logger.info(f"Stop loss triggered at {current_price} (entry: {avg_entry})")
                return "SELL", abs(self.current_position)
            
            # Check take profit
            take_profit = avg_entry * (1 + self.config["take_profit_pct"])
            if current_price >= take_profit:
                logger.info(f"Take profit triggered at {current_price} (entry: {avg_entry})")
                return "SELL", abs(self.current_position)
        
        elif self.current_position < 0 and short_trades:
            # We have a short position
            avg_entry = sum(t["price"] * (t["position_before"] - t["position_after"]) for t in short_trades) / abs(self.current_position)
            
            # Check stop loss
            stop_loss = avg_entry * (1 + self.config["stop_loss_pct"])
            if current_price >= stop_loss:
                logger.info(f"Stop loss triggered at {current_price} (entry: {avg_entry})")
                return "BUY", abs(self.current_position)
            
            # Check take profit
            take_profit = avg_entry * (1 - self.config["take_profit_pct"])
            if current_price <= take_profit:
                logger.info(f"Take profit triggered at {current_price} (entry: {avg_entry})")
                return "BUY", abs(self.current_position)
        
        return None, 0
    
    def update_data(self, new_data):
        """
        Update the data buffer with new market data.
        
        Parameters:
        -----------
        new_data : pandas.DataFrame or dict
            New market data to add to the buffer.
        """
        if isinstance(new_data, dict):
            # Convert dict to DataFrame
            new_data = pd.DataFrame([new_data])
        
        # Ensure timestamp column exists and is datetime
        if 'Timestamp' in new_data.columns and not pd.api.types.is_datetime64_any_dtype(new_data['Timestamp']):
            new_data['Timestamp'] = pd.to_datetime(new_data['Timestamp'])
        
        # Set timestamp as index if not already
        if 'Timestamp' in new_data.columns:
            new_data = new_data.set_index('Timestamp')
        
        # Update buffer
        if self.data_buffer.empty:
            self.data_buffer = new_data
        else:
            self.data_buffer = pd.concat([self.data_buffer, new_data])
        
        # Trim buffer to configured size
        buffer_size = self.config["buffer_size"]
        if len(self.data_buffer) > buffer_size:
            self.data_buffer = self.data_buffer.iloc[-buffer_size:]
        
        logger.debug(f"Updated data buffer, new size: {len(self.data_buffer)}")
    
    def process_tick(self, tick_data):
        """
        Process a new market tick.
        
        Parameters:
        -----------
        tick_data : dict
            Dict containing tick data with at least 'Timestamp' and 'Close' keys.
            
        Returns:
        --------
        action : str or None
            Trading action to take, if any.
        """
        # Update data buffer
        self.update_data(tick_data)
        
        # Check if we have enough data
        if len(self.data_buffer) < 50:  # Need some minimum data for feature engineering
            logger.warning("Not enough data in buffer")
            return None
        
        # Check if we should generate a signal
        current_time = pd.to_datetime(tick_data.get('Timestamp', datetime.now()))
        if not self._should_generate_signal(current_time):
            return None
        
        # Engineer features
        try:
            features = self._engineer_features(self.data_buffer)
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return None
        
        if features.empty:
            logger.warning("No features after engineering")
            return None
        
        # Check stop loss / take profit
        current_price = tick_data.get('Close')
        if current_price is not None:
            sl_tp_action, sl_tp_size = self._check_stop_loss_take_profit(current_price)
            
            if sl_tp_action and sl_tp_size > 0:
                # Execute stop loss or take profit
                signal = 1 if sl_tp_action == "BUY" else -1
                success = self._execute_trade(signal, current_price, current_time, 1.0)
                
                if success:
                    return sl_tp_action
        
        # Generate signal
        signal, confidence = self._generate_signal(features)
        
        # Execute trade based on signal
        if signal != 0:
            success = self._execute_trade(signal, current_price, current_time, confidence)
            
            if success:
                return "BUY" if signal > 0 else "SELL"
        
        return None
    
    def get_position(self):
        """
        Get current position.
        
        Returns:
        --------
        position : int
            Current position.
        """
        return self.current_position
    
    def get_trades(self):
        """
        Get trade history.
        
        Returns:
        --------
        trades : list
            List of trades.
        """
        return self.trades
    
    def save_state(self, path='trading_system_state.json'):
        """
        Save the current state of the trading system.
        
        Parameters:
        -----------
        path : str
            Path to save the state.
        """
        state = {
            "current_position": self.current_position,
            "trades": self.trades,
            "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"State saved to {path}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self, path='trading_system_state.json'):
        """
        Load the state of the trading system.
        
        Parameters:
        -----------
        path : str
            Path to load the state from.
        """
        if not os.path.exists(path):
            logger.warning(f"State file {path} does not exist")
            return
        
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            self.current_position = state.get("current_position", 0)
            self.trades = state.get("trades", [])
            
            last_signal_time = state.get("last_signal_time")
            if last_signal_time:
                self.last_signal_time = datetime.fromisoformat(last_signal_time)
            
            logger.info(f"State loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading state: {e}")

def simulate_trading(trading_system, data_file, output_file=None):
    """
    Simulate trading using historical data.
    
    Parameters:
    -----------
    trading_system : ESFuturesTradingSystem
        Trading system instance.
    data_file : str
        Path to historical data file.
    output_file : str, optional
        Path to save simulation results.
    """
    # Load historical data
    df = pd.read_csv(data_file)
    
    # Ensure timestamp column exists and is datetime
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Sort by timestamp
    if 'Timestamp' in df.columns:
        df = df.sort_values('Timestamp')
    
    # Initialize results
    results = []
    
    # Simulate tick by tick
    for _, row in df.iterrows():
        tick_data = row.to_dict()
        
        # Process tick
        action = trading_system.process_tick(tick_data)
        
        # Store result
        result = {
            'Timestamp': tick_data.get('Timestamp'),
            'Price': tick_data.get('Close'),
            'Action': action,
            'Position': trading_system.get_position()
        }
        
        results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results if output file specified
    if output_file:
        results_df.to_csv(output_file, index=False)
        logger.info(f"Simulation results saved to {output_file}")
    
    # Print summary
    trades = trading_system.get_trades()
    
    print("\nTrading Simulation Summary")
    print("==========================")
    print(f"Total ticks processed: {len(df)}")
    print(f"Total trades executed: {len(trades)}")
    print(f"Final position: {trading_system.get_position()}")
    
    # Calculate P&L if we have trades
    if trades:
        buy_trades = [t for t in trades if t["action"] == "BUY"]
        sell_trades = [t for t in trades if t["action"] == "SELL"]
        
        total_bought = sum(t["price"] * t["size"] for t in buy_trades)
        total_sold = sum(t["price"] * t["size"] for t in sell_trades)
        
        # Adjust for any open position
        current_position = trading_system.get_position()
        last_price = df['Close'].iloc[-1] if 'Close' in df.columns else None
        
        if current_position != 0 and last_price is not None:
            if current_position > 0:
                # Have long position, simulate selling at last price
                total_sold += current_position * last_price
            else:
                # Have short position, simulate buying at last price
                total_bought += abs(current_position) * last_price
        
        pnl = total_sold - total_bought
        
        print(f"Estimated P&L: ${pnl:.2f}")
    
    return results_df

def main():
    """
    Main function to run the trading system.
    """
    parser = argparse.ArgumentParser(description='ES Futures Trading System')
    parser.add_argument('--config', type=str, default='trading-config.json', help='Path to configuration file')
    parser.add_argument('--simulate', type=str, help='Path to historical data file for simulation')
    parser.add_argument('--output', type=str, help='Path to save simulation results')

    # Use parse_known_args() to avoid conflicts with Jupyter arguments
    args, unknown = parser.parse_known_args()

    # Initialize trading system
    trading_system = ESFuturesTradingSystem(config_path=args.config)
    
    # Run in simulation mode if specified
    if args.simulate:
        logger.info(f"Running simulation with data from {args.simulate}")
        simulate_trading(trading_system, args.simulate, args.output)
    else:
        logger.info("Trading system ready")
        
        # In a real-world scenario, we would connect to a trading API
        # and process real-time market data
        print("Trading system initialized. In a real-world scenario, ")
        print("this would connect to a trading API for real-time data.")
        
        # Simple command loop for demo purposes
        while True:
            cmd = input("\nEnter command (quit, position, trades, save, load): ")
            
            if cmd.lower() == 'quit':
                break
            elif cmd.lower() == 'position':
                print(f"Current position: {trading_system.get_position()}")
            elif cmd.lower() == 'trades':
                trades = trading_system.get_trades()
                print(f"Total trades: {len(trades)}")
                for i, trade in enumerate(trades[-5:]):  # Show last 5 trades
                    print(f"{i+1}. {trade['timestamp']} - {trade['action']} {trade['size']} @ {trade['price']}")
            elif cmd.lower() == 'save':
                trading_system.save_state()
            elif cmd.lower() == 'load':
                trading_system.load_state()
            else:
                print("Unknown command")

if __name__ == "__main__":
    main()