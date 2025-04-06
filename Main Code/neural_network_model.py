import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

class DeepLearningModels:
    """
    Class containing various deep learning models for time series forecasting
    of ES futures prices.
    """
    
    def __init__(self, lookback=20):
        """
        Initialize the deep learning models class.
        
        Parameters:
        -----------
        lookback : int
            Number of time steps to look back for sequence models.
        """
        self.lookback = lookback
        self.models = {}
        self.histories = {}
        self.scalers = {}
    
    def prepare_sequence_data(self, X, y, lookback=None):
        """
        Convert input data into sequences for LSTM/GRU models.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix.
        y : numpy.ndarray
            Target vector.
        lookback : int, optional
            Number of time steps to look back.
            
        Returns:
        --------
        X_seq : numpy.ndarray
            Sequence data with shape (n_samples, lookback, n_features).
        y_seq : numpy.ndarray
            Target values corresponding to the sequences.
        """
        if lookback is None:
            lookback = self.lookback
        
        n_samples, n_features = X.shape
        X_seq = []
        y_seq = []
        
        for i in range(lookback, n_samples):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_lstm_model(self, input_shape, output_shape=1, lstm_units=[50, 50], dropout_rate=0.2):
        """
        Build a stacked LSTM model.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input sequences (lookback, n_features).
        output_shape : int
            Number of output units.
        lstm_units : list
            List of LSTM units for each layer.
        dropout_rate : float
            Dropout rate for regularization.
            
        Returns:
        --------
        model : tensorflow.keras.models.Sequential
            Compiled LSTM model.
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=lstm_units[0],
            return_sequences=len(lstm_units) > 1,
            input_shape=input_shape
        ))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        
        # Additional LSTM layers
        for i in range(1, len(lstm_units)):
            model.add(LSTM(
                units=lstm_units[i],
                return_sequences=i < len(lstm_units) - 1
            ))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
        
        # Output layer
        model.add(Dense(output_shape))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )
        
        return model
    
    def build_gru_model(self, input_shape, output_shape=1, gru_units=[64, 32], dropout_rate=0.2):
        """
        Build a stacked GRU model.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input sequences (lookback, n_features).
        output_shape : int
            Number of output units.
        gru_units : list
            List of GRU units for each layer.
        dropout_rate : float
            Dropout rate for regularization.
            
        Returns:
        --------
        model : tensorflow.keras.models.Sequential
            Compiled GRU model.
        """
        model = Sequential()
        
        # First GRU layer
        model.add(GRU(
            units=gru_units[0],
            return_sequences=len(gru_units) > 1,
            input_shape=input_shape
        ))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        
        # Additional GRU layers
        for i in range(1, len(gru_units)):
            model.add(GRU(
                units=gru_units[i],
                return_sequences=i < len(gru_units) - 1
            ))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
        
        # Output layer
        model.add(Dense(output_shape))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )
        
        return model
    
    def build_hybrid_model(self, seq_input_shape, aux_input_shape, output_shape=1):
        """
        Build a hybrid model combining sequential data with auxiliary features.
        
        Parameters:
        -----------
        seq_input_shape : tuple
            Shape of sequential input (lookback, n_seq_features).
        aux_input_shape : tuple
            Shape of auxiliary input (n_aux_features,).
        output_shape : int
            Number of output units.
            
        Returns:
        --------
        model : tensorflow.keras.models.Model
            Compiled hybrid model.
        """
        # Sequential input branch
        seq_input = Input(shape=seq_input_shape)
        x1 = LSTM(64, return_sequences=True)(seq_input)
        x1 = Dropout(0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = LSTM(32)(x1)
        x1 = Dropout(0.2)(x1)
        x1 = BatchNormalization()(x1)
        
        # Auxiliary input branch
        aux_input = Input(shape=aux_input_shape)
        x2 = Dense(32, activation='relu')(aux_input)
        x2 = Dropout(0.2)(x2)
        x2 = BatchNormalization()(x2)
        
        # Combine branches
        combined = Concatenate()([x1, x2])
        
        # Output layers
        x = Dense(32, activation='relu')(combined)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        output = Dense(output_shape)(x)
        
        # Create model
        model = Model(inputs=[seq_input, aux_input], outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )
        
        return model
    
    def train_lstm_model(self, X_train, y_train, X_val=None, y_val=None, 
                         epochs=100, batch_size=32, verbose=1):
        """
        Train an LSTM model on the given data.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features.
        y_train : numpy.ndarray
            Training targets.
        X_val : numpy.ndarray, optional
            Validation features.
        y_val : numpy.ndarray, optional
            Validation targets.
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size for training.
        verbose : int
            Verbosity level.
            
        Returns:
        --------
        model : tensorflow.keras.models.Sequential
            Trained LSTM model.
        history : tensorflow.keras.callbacks.History
            Training history.
        """
        # Scale the data
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = scaler_X.transform(X_val)
        
        # Prepare sequence data
        X_train_seq, y_train_seq = self.prepare_sequence_data(X_train_scaled, y_train)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_sequence_data(X_val_scaled, y_val)
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
        
        # Build model
        model = self.build_lstm_model(input_shape=(self.lookback, X_train.shape[1]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
        ]
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Store model and history
        self.models['lstm'] = model
        self.histories['lstm'] = history
        self.scalers['lstm'] = scaler_X
        
        return model, history
    
    def train_gru_model(self, X_train, y_train, X_val=None, y_val=None, 
                        epochs=100, batch_size=32, verbose=1):
        """
        Train a GRU model on the given data.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features.
        y_train : numpy.ndarray
            Training targets.
        X_val : numpy.ndarray, optional
            Validation features.
        y_val : numpy.ndarray, optional
            Validation targets.
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size for training.
        verbose : int
            Verbosity level.
            
        Returns:
        --------
        model : tensorflow.keras.models.Sequential
            Trained GRU model.
        history : tensorflow.keras.callbacks.History
            Training history.
        """
        # Scale the data
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = scaler_X.transform(X_val)
        
        # Prepare sequence data
        X_train_seq, y_train_seq = self.prepare_sequence_data(X_train_scaled, y_train)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_sequence_data(X_val_scaled, y_val)
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
        
        # Build model
        model = self.build_gru_model(input_shape=(self.lookback, X_train.shape[1]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
        ]
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Store model and history
        self.models['gru'] = model
        self.histories['gru'] = history
        self.scalers['gru'] = scaler_X
        
        return model, history
    
    def train_hybrid_model(self, X_train_seq, X_train_aux, y_train,
                           X_val_seq=None, X_val_aux=None, y_val=None,
                           epochs=100, batch_size=32, verbose=1):
        """
        Train a hybrid model on the given data.
        
        Parameters:
        -----------
        X_train_seq : numpy.ndarray
            Sequential training features.
        X_train_aux : numpy.ndarray
            Auxiliary training features.
        y_train : numpy.ndarray
            Training targets.
        X_val_seq : numpy.ndarray, optional
            Sequential validation features.
        X_val_aux : numpy.ndarray, optional
            Auxiliary validation features.
        y_val : numpy.ndarray, optional
            Validation targets.
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size for training.
        verbose : int
            Verbosity level.
            
        Returns:
        --------
        model : tensorflow.keras.models.Model
            Trained hybrid model.
        history : tensorflow.keras.callbacks.History
            Training history.
        """
        # Scale the data
        scaler_seq = StandardScaler()
        scaler_aux = StandardScaler()
        
        X_train_seq_scaled = scaler_seq.fit_transform(X_train_seq)
        X_train_aux_scaled = scaler_aux.fit_transform(X_train_aux)
        
        if X_val_seq is not None and X_val_aux is not None:
            X_val_seq_scaled = scaler_seq.transform(X_val_seq)
            X_val_aux_scaled = scaler_aux.transform(X_val_aux)
        
        # Prepare sequence data
        X_train_seq_reshaped, y_train_reshaped = self.prepare_sequence_data(X_train_seq_scaled, y_train)
        
        if X_val_seq is not None and X_val_aux is not None and y_val is not None:
            X_val_seq_reshaped, y_val_reshaped = self.prepare_sequence_data(X_val_seq_scaled, y_val)
            
            # For the auxiliary data, we need to match the sequence length
            X_val_aux_aligned = X_val_aux_scaled[self.lookback:len(X_val_aux_scaled)]
            
            validation_data = ([X_val_seq_reshaped, X_val_aux_aligned], y_val_reshaped)
        else:
            validation_data = None
        
        # For the auxiliary data, we need to match the sequence length
        X_train_aux_aligned = X_train_aux_scaled[self.lookback:len(X_train_aux_scaled)]
        
        # Build model
        model = self.build_hybrid_model(
            seq_input_shape=(self.lookback, X_train_seq.shape[1]),
            aux_input_shape=(X_train_aux.shape[1],)
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
        ]
        
        # Train model
        history = model.fit(
            [X_train_seq_reshaped, X_train_aux_aligned], y_train_reshaped,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Store model and history
        self.models['hybrid'] = model
        self.histories['hybrid'] = history
        self.scalers['hybrid_seq'] = scaler_seq
        self.scalers['hybrid_aux'] = scaler_aux
        
        return model, history
    
    def predict_lstm(self, X_test):
        """
        Generate predictions using the trained LSTM model.
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test features.
            
        Returns:
        --------
        y_pred : numpy.ndarray
            Predicted values.
        """
        if 'lstm' not in self.models:
            raise ValueError("LSTM model has not been trained yet.")
        
        # Scale the data
        X_test_scaled = self.scalers['lstm'].transform(X_test)
        
        # Prepare sequence data
        X_test_seq = []
        for i in range(self.lookback, len(X_test_scaled)):
            X_test_seq.append(X_test_scaled[i-self.lookback:i])
        
        X_test_seq = np.array(X_test_seq)
        
        # Generate predictions
        y_pred = self.models['lstm'].predict(X_test_seq)
        
        return y_pred.flatten()
    
    def predict_gru(self, X_test):
        """
        Generate predictions using the trained GRU model.
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test features.
            
        Returns:
        --------
        y_pred : numpy.ndarray
            Predicted values.
        """
        if 'gru' not in self.models:
            raise ValueError("GRU model has not been trained yet.")
        
        # Scale the data
        X_test_scaled = self.scalers['gru'].transform(X_test)
        
        # Prepare sequence data
        X_test_seq = []
        for i in range(self.lookback, len(X_test_scaled)):
            X_test_seq.append(X_test_scaled[i-self.lookback:i])
        
        X_test_seq = np.array(X_test_seq)
        
        # Generate predictions
        y_pred = self.models['gru'].predict(X_test_seq)
        
        return y_pred.flatten()
    
    def predict_hybrid(self, X_test_seq, X_test_aux):
        """
        Generate predictions using the trained hybrid model.
        
        Parameters:
        -----------
        X_test_seq : numpy.ndarray
            Sequential test features.
        X_test_aux : numpy.ndarray
            Auxiliary test features.
            
        Returns:
        --------
        y_pred : numpy.ndarray
            Predicted values.
        """
        if 'hybrid' not in self.models:
            raise ValueError("Hybrid model has not been trained yet.")
        
        # Scale the data
        X_test_seq_scaled = self.scalers['hybrid_seq'].transform(X_test_seq)
        X_test_aux_scaled = self.scalers['hybrid_aux'].transform(X_test_aux)
        
        # Prepare sequence data
        X_test_seq_reshaped = []
        for i in range(self.lookback, len(X_test_seq_scaled)):
            X_test_seq_reshaped.append(X_test_seq_scaled[i-self.lookback:i])
        
        X_test_seq_reshaped = np.array(X_test_seq_reshaped)
        
        # For the auxiliary data, we need to match the sequence length
        X_test_aux_aligned = X_test_aux_scaled[self.lookback:len(X_test_aux_scaled)]
        
        # Generate predictions
        y_pred = self.models['hybrid'].predict([X_test_seq_reshaped, X_test_aux_aligned])
        
        return y_pred.flatten()
    
    def plot_training_history(self, model_name):
        """
        Plot the training history for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to plot history for.
        """
        if model_name not in self.histories:
            raise ValueError(f"No training history available for {model_name} model.")
        
        history = self.histories[model_name]
        
        plt.figure(figsize=(12, 5))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'])
            plt.legend(['Train', 'Validation'])
        else:
            plt.legend(['Train'])
        
        plt.title(f'{model_name.upper()} Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot learning rate if available
        if 'lr' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_training_history.png')
        plt.close()

def select_features(df, n_features=None, correlation_threshold=0.7):
    """
    Select the most relevant features based on correlation with the target.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing features and target.
    n_features : int, optional
        Number of features to select. If None, use correlation threshold.
    correlation_threshold : float, optional
        Correlation threshold for feature selection.
        
    Returns:
    --------
    selected_features : list
        List of selected feature names.
    """
    # Assume the future return column is the target
    target_col = [col for col in df.columns if col.startswith('future_return_')][0]
    
    # Calculate correlation with target
    correlations = df.corr()[target_col].abs().sort_values(ascending=False)
    
    # Remove target and other future-related columns
    correlations = correlations.drop([col for col in correlations.index if col.startswith('future_')])
    
    if n_features is not None:
        # Select top n features
        selected_features = correlations.head(n_features).index.tolist()
    else:
        # Select features above correlation threshold
        selected_features = correlations[correlations > correlation_threshold].index.tolist()
    
    return selected_features

def split_features(df, seq_features=None, aux_features=None):
    """
    Split features into sequential and auxiliary features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing features.
    seq_features : list, optional
        List of sequential feature names.
    aux_features : list, optional
        List of auxiliary feature names.
        
    Returns:
    --------
    X_seq : numpy.ndarray
        Sequential features.
    X_aux : numpy.ndarray
        Auxiliary features.
    """
    if seq_features is None:
        # Use price and volume related features as sequential
        price_cols = ['Open', 'High', 'Low', 'Close']
        volume_cols = [col for col in df.columns if 'Volume' in col]
        ma_cols = [col for col in df.columns if col.startswith('ma_') or col.startswith('ema_')]
        
        seq_features = price_cols + volume_cols + ma_cols
    
    if aux_features is None:
        # Use all other numerical features as auxiliary
        exclude_cols = seq_features + [col for col in df.columns if col.startswith('future_')]
        aux_features = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.int64, np.float64]]
    
    X_seq = df[seq_features].values
    X_aux = df[aux_features].values
    
    return X_seq, X_aux, seq_features, aux_features

def train_neural_networks(df, target_col='future_return_10', test_size=0.2, lookback=20):
    """
    Train various neural network models on the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing features and target.
    target_col : str
        Target column for prediction.
    test_size : float
        Proportion of data to use for testing.
    lookback : int
        Number of time steps to look back for sequence models.
        
    Returns:
    --------
    dl_models : DeepLearningModels
        Trained deep learning models.
    results : dict
        Dictionary containing predictions and model evaluations.
    """
    # Split data into train and test sets
    train_size = int(len(df) * (1 - test_size))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Extract features and target
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    
    # Split features into sequential and auxiliary
    X_train_seq, X_train_aux, seq_features, aux_features = split_features(train_df)
    X_test_seq, X_test_aux, _, _ = split_features(test_df, seq_features, aux_features)
    
    # Initialize deep learning models
    dl_models = DeepLearningModels(lookback=lookback)
    
    # Train LSTM model
    print("Training LSTM model...")
    lstm_model, lstm_history = dl_models.train_lstm_model(
        X_train_seq, y_train,
        epochs=50, batch_size=32, verbose=1
    )
    
    # Train GRU model
    print("Training GRU model...")
    gru_model, gru_history = dl_models.train_gru_model(
        X_train_seq, y_train,
        epochs=50, batch_size=32, verbose=1
    )
    
    # Train hybrid model
    print("Training hybrid model...")
    hybrid_model, hybrid_history = dl_models.train_hybrid_model(
        X_train_seq, X_train_aux, y_train,
        epochs=50, batch_size=32, verbose=1
    )
    
    # Generate predictions
    print("Generating predictions...")
    lstm_preds = dl_models.predict_lstm(X_test_seq)
    gru_preds = dl_models.predict_gru(X_test_seq)
    hybrid_preds = dl_models.predict_hybrid(X_test_seq, X_test_aux)
    
    # Evaluate models
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = {
        'lstm': {
            'predictions': lstm_preds,
            'mse': mean_squared_error(y_test[lookback:], lstm_preds),
            'mae': mean_absolute_error(y_test[lookback:], lstm_preds),
            'r2': r2_score(y_test[lookback:], lstm_preds)
        },
        'gru': {
            'predictions': gru_preds,
            'mse': mean_squared_error(y_test[lookback:], gru_preds),
            'mae': mean_absolute_error(y_test[lookback:], gru_preds),
            'r2': r2_score(y_test[lookback:], gru_preds)
        },
        'hybrid': {
            'predictions': hybrid_preds,
            'mse': mean_squared_error(y_test[lookback:], hybrid_preds),
            'mae': mean_absolute_error(y_test[lookback:], hybrid_preds),
            'r2': r2_score(y_test[lookback:], hybrid_preds)
        }
    }
    
    # Plot training history
    dl_models.plot_training_history('lstm')
    dl_models.plot_training_history('gru')
    dl_models.plot_training_history('hybrid')
    
    # Print evaluation results
    print("\n=== Neural Network Model Evaluation ===")
    for model_name, metrics in results.items():
        print(f"{model_name.upper()} - MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, R²: {metrics['r2']:.6f}")
    
    return dl_models, results

def generate_nn_trading_signals(test_df, results, lookback=20, threshold=0.0001):
    """
    Generate trading signals based on neural network predictions.
    
    Parameters:
    -----------
    test_df : pandas.DataFrame
        Test dataset.
    results : dict
        Dictionary containing model predictions.
    lookback : int
        Number of time steps used for lookback in sequence models.
    threshold : float
        Threshold for generating buy/sell signals.
        
    Returns:
    --------
    signals_df : pandas.DataFrame
        DataFrame containing trading signals.
    """
    # Create a copy of the test dataframe for signals
    signals_df = test_df.copy()
    
    # Add predictions to the signals dataframe
    for model_name, metrics in results.items():
        # Skip the first 'lookback' rows as they don't have predictions
        signals_df[f'{model_name}_pred'] = np.nan
        signals_df.iloc[lookback:, signals_df.columns.get_loc(f'{model_name}_pred')] = metrics['predictions']
        
        # Generate trading signals
        signals_df[f'{model_name}_position'] = np.where(
            signals_df[f'{model_name}_pred'] > threshold, 1,  # Buy
            np.where(signals_df[f'{model_name}_pred'] < -threshold, -1, 0)  # Sell or Hold
        )
    
    # Generate ensemble signal
    model_preds = []
    for model_name in results.keys():
        pred_col = f'{model_name}_pred'
        if pred_col in signals_df.columns:
            model_preds.append(signals_df[pred_col])
    
    if model_preds:
        signals_df['ensemble_nn_pred'] = sum(model_preds) / len(model_preds)
        signals_df['ensemble_nn_position'] = np.where(
            signals_df['ensemble_nn_pred'] > threshold, 1,  # Buy
            np.where(signals_df['ensemble_nn_pred'] < -threshold, -1, 0)  # Sell or Hold
        )
    
    return signals_df