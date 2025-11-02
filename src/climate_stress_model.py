"""
Advanced Climate Stress Projection Models using Deep Learning
State-of-the-art neural networks for climate impact analysis and prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ClimateStressModeler:
    """
    Advanced climate stress modeling using ensemble of deep learning architectures
    including CNNs, LSTMs, Transformers, and attention mechanisms
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.climate_projections = {}
        self.stress_indices = {}
        
    def build_climate_cnn_lstm(self, input_shape):
        """Build CNN-LSTM hybrid model for spatiotemporal climate data"""
        # CNN feature extraction
        cnn_input = layers.Input(shape=input_shape)
        cnn_layers = []
        
        # Multiple CNN scales
        for filters in [32, 64, 128]:
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(cnn_input)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.2)(x)
            cnn_layers.append(x)
        
        # Concatenate multi-scale features
        cnn_features = layers.Concatenate()(cnn_layers)
        cnn_features = layers.GlobalAveragePooling2D()(cnn_features)
        
        # Reshape for LSTM
        cnn_features = layers.Reshape((1, cnn_features.shape[-1]))(cnn_features)
        
        # LSTM layers
        lstm_out = layers.LSTM(128, return_sequences=True)(cnn_features)
        lstm_out = layers.BatchNormalization()(lstm_out)
        lstm_out = layers.Dropout(0.2)(lstm_out)
        
        lstm_out = layers.LSTM(64, return_sequences=False)(lstm_out)
        lstm_out = layers.BatchNormalization()(lstm_out)
        lstm_out = layers.Dropout(0.2)(lstm_out)
        
        # Dense layers
        dense = layers.Dense(128, activation='relu')(lstm_out)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dropout(0.3)(dense)
        
        dense = layers.Dense(64, activation='relu')(dense)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dropout(0.2)(dense)
        
        # Output layer - predict multiple climate stress indicators
        outputs = layers.Dense(5, activation='linear')(dense)  # [temp_stress, precip_stress, drought_risk, flood_risk, combined_stress]
        
        model = keras.Model(inputs=cnn_input, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def build_climate_transformer(self, sequence_length, feature_dim):
        """Build Transformer model for climate sequence prediction"""
        inputs = layers.Input(shape=(sequence_length, feature_dim))
        
        # Positional encoding
        positions = tf.range(start=0, limit=sequence_length, delta=1)
        position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=feature_dim)(positions)
        
        # Add positional encoding
        x = inputs + position_embeddings
        
        # Multiple transformer blocks
        for _ in range(4):
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=8, key_dim=feature_dim
            )(x, x)
            
            # Add & Norm
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization()(x)
            
            # Feed forward
            ff_output = layers.Dense(feature_dim * 4, activation='relu')(x)
            ff_output = layers.Dense(feature_dim)(ff_output)
            
            # Add & Norm
            x = layers.Add()([x, ff_output])
            x = layers.LayerNormalization()(x)
        
        # Global attention pooling
        attention_weights = layers.Dense(1, activation='softmax')(x)
        pooled_output = tf.reduce_sum(x * attention_weights, axis=1)
        
        # Dense layers
        dense = layers.Dense(256, activation='relu')(pooled_output)
        dense = layers.Dropout(0.3)(dense)
        
        dense = layers.Dense(128, activation='relu')(dense)
        dense = layers.Dropout(0.2)(dense)
        
        # Multi-task outputs
        temp_stress = layers.Dense(1, activation='linear', name='temp_stress')(dense)
        precip_stress = layers.Dense(1, activation='linear', name='precip_stress')(dense)
        drought_risk = layers.Dense(1, activation='sigmoid', name='drought_risk')(dense)
        flood_risk = layers.Dense(1, activation='sigmoid', name='flood_risk')(dense)
        
        model = keras.Model(
            inputs=inputs, 
            outputs=[temp_stress, precip_stress, drought_risk, flood_risk]
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'temp_stress': 'mse',
                'precip_stress': 'mse', 
                'drought_risk': 'binary_crossentropy',
                'flood_risk': 'binary_crossentropy'
            },
            loss_weights={
                'temp_stress': 0.3,
                'precip_stress': 0.3,
                'drought_risk': 0.2,
                'flood_risk': 0.2
            },
            metrics={
                'temp_stress': ['mae'],
                'precip_stress': ['mae'],
                'drought_risk': ['accuracy'],
                'flood_risk': ['accuracy']
            }
        )
        
        return model
    
    def build_climate_gan(self, latent_dim, climate_dim):
        """Build Generative Adversarial Network for climate scenario generation"""
        # Generator
        generator_input = layers.Input(shape=(latent_dim,))
        
        gen = layers.Dense(128, activation='relu')(generator_input)
        gen = layers.BatchNormalization()(gen)
        gen = layers.LeakyReLU(0.2)(gen)
        
        gen = layers.Dense(256, activation='relu')(gen)
        gen = layers.BatchNormalization()(gen)
        gen = layers.LeakyReLU(0.2)(gen)
        
        gen = layers.Dense(512, activation='relu')(gen)
        gen = layers.BatchNormalization()(gen)
        gen = layers.LeakyReLU(0.2)(gen)
        
        generator_output = layers.Dense(climate_dim, activation='tanh')(gen)
        
        generator = keras.Model(generator_input, generator_output)
        
        # Discriminator
        discriminator_input = layers.Input(shape=(climate_dim,))
        
        disc = layers.Dense(512, activation='relu')(discriminator_input)
        disc = layers.LeakyReLU(0.2)(disc)
        disc = layers.Dropout(0.3)(disc)
        
        disc = layers.Dense(256, activation='relu')(disc)
        disc = layers.LeakyReLU(0.2)(disc)
        disc = layers.Dropout(0.3)(disc)
        
        disc = layers.Dense(128, activation='relu')(disc)
        disc = layers.LeakyReLU(0.2)(disc)
        disc = layers.Dropout(0.2)(disc)
        
        discriminator_output = layers.Dense(1, activation='sigmoid')(disc)
        
        discriminator = keras.Model(discriminator_input, discriminator_output)
        discriminator.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Combined GAN
        discriminator.trainable = False
        gan_output = discriminator(generator_output)
        gan = keras.Model(generator_input, gan_output)
        gan.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )
        
        return generator, discriminator, gan
    
    def calculate_climate_stress_indices(self, climate_data):
        """Calculate comprehensive climate stress indices"""
        stress_indices = {}
        
        if isinstance(climate_data, pd.DataFrame):
            # Temperature stress index
            if 'temperature' in climate_data.columns:
                temp_mean = climate_data['temperature'].mean()
                temp_std = climate_data['temperature'].std()
                temp_stress = np.abs(climate_data['temperature'] - temp_mean) / (temp_std + 1e-8)
                stress_indices['temperature_stress'] = temp_stress
            
            # Precipitation stress index
            if 'precipitation' in climate_data.columns:
                precip_cv = climate_data['precipitation'].std() / (climate_data['precipitation'].mean() + 1e-8)
                precip_stress = np.abs(climate_data['precipitation'] - climate_data['precipitation'].mean()) / (climate_data['precipitation'].mean() + 1e-8)
                stress_indices['precipitation_stress'] = precip_stress
                stress_indices['precipitation_variability'] = precip_cv
            
            # Drought index (Standardized Precipitation Evapotranspiration Index)
            if 'precipitation' in climate_data.columns and 'temperature' in climate_data.columns:
                # Simplified SPEI calculation
                pet = 0.0023 * (climate_data['temperature'] + 17.8) * (climate_data['temperature'] - 17.8) ** 0.5  # Hargreaves PET
                water_balance = climate_data['precipitation'] - pet
                spei = (water_balance - water_balance.mean()) / (water_balance.std() + 1e-8)
                stress_indices['drought_index'] = -spei  # Negative for drought stress
            
            # Heat stress index
            if 'temperature_max' in climate_data.columns:
                heat_stress_days = (climate_data['temperature_max'] > 35).astype(int)
                stress_indices['heat_stress_frequency'] = heat_stress_days.rolling(window=30).sum()
            
            # Flood risk index
            if 'precipitation' in climate_data.columns:
                extreme_precip = climate_data['precipitation'] > climate_data['precipitation'].quantile(0.95)
                flood_risk = extreme_precip.rolling(window=7).sum()  # 7-day extreme precipitation
                stress_indices['flood_risk'] = flood_risk
            
            # Combined climate stress index
            available_indices = [v for v in stress_indices.values() if isinstance(v, (pd.Series, np.ndarray))]
            if available_indices:
                combined_stress = np.mean(available_indices, axis=0)
                stress_indices['combined_stress'] = combined_stress
        
        self.stress_indices = stress_indices
        return stress_indices
    
    def predict_future_climate_stress(self, historical_data, years_ahead=10):
        """Predict future climate stress using advanced time series models"""
        predictions = {}
        
        if isinstance(historical_data, pd.DataFrame):
            # Prepare sequences for time series prediction
            sequence_length = 12  # 12 months
            future_predictions = {}
            
            for col in historical_data.select_dtypes(include=[np.number]).columns:
                data = historical_data[col].values
                
                # Create sequences
                X, y = [], []
                for i in range(len(data) - sequence_length):
                    X.append(data[i:i+sequence_length])
                    y.append(data[i+sequence_length])
                
                X = np.array(X)
                y = np.array(y)
                
                if len(X) > 0:
                    # Build LSTM for prediction
                    model = models.Sequential([
                        layers.LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
                        layers.Dense(25, activation='relu'),
                        layers.Dense(1)
                    ])
                    
                    model.compile(optimizer='adam', loss='mse')
                    
                    # Train model
                    X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
                    model.fit(X_reshaped, y, epochs=100, verbose=0)
                    
                    # Make future predictions
                    future_seq = data[-sequence_length:].reshape(1, sequence_length, 1)
                    future_vals = []
                    
                    for _ in range(years_ahead * 12):  # Monthly predictions
                        next_val = model.predict(future_seq, verbose=0)[0, 0]
                        future_vals.append(next_val)
                        
                        # Update sequence
                        future_seq = np.roll(future_seq, -1, axis=1)
                        future_seq[0, -1, 0] = next_val
                    
                    future_predictions[col] = future_vals
            
            predictions['future_projections'] = future_predictions
            
            # Calculate future stress indices
            future_df = pd.DataFrame(future_predictions)
            future_stress = self.calculate_climate_stress_indices(future_df)
            predictions['future_stress_indices'] = future_stress
        
        return predictions
    
    def identify_climate_whiplash_events(self, climate_data):
        """Identify climate whiplash events (rapid transitions between extremes)"""
        whiplash_events = []
        
        if isinstance(climate_data, pd.DataFrame):
            if 'precipitation' in climate_data.columns and 'temperature' in climate_data.columns:
                # Calculate precipitation percentiles
                precip_p10 = climate_data['precipitation'].quantile(0.1)
                precip_p90 = climate_data['precipitation'].quantile(0.9)
                
                # Calculate temperature percentiles
                temp_p10 = climate_data['temperature'].quantile(0.1)
                temp_p90 = climate_data['temperature'].quantile(0.9)
                
                # Identify whiplash events
                for i in range(1, len(climate_data)):
                    # Dry to wet whiplash
                    if (climate_data['precipitation'].iloc[i-1] <= precip_p10 and 
                        climate_data['precipitation'].iloc[i] >= precip_p90):
                        whiplash_events.append({
                            'date': climate_data.index[i],
                            'type': 'dry_to_wet',
                            'precipitation_change': climate_data['precipitation'].iloc[i] - climate_data['precipitation'].iloc[i-1],
                            'severity': 'extreme'
                        })
                    
                    # Wet to dry whiplash
                    elif (climate_data['precipitation'].iloc[i-1] >= precip_p90 and 
                          climate_data['precipitation'].iloc[i] <= precip_p10):
                        whiplash_events.append({
                            'date': climate_data.index[i],
                            'type': 'wet_to_dry',
                            'precipitation_change': climate_data['precipitation'].iloc[i-1] - climate_data['precipitation'].iloc[i],
                            'severity': 'extreme'
                        })
                    
                    # Cold to hot whiplash
                    elif (climate_data['temperature'].iloc[i-1] <= temp_p10 and 
                          climate_data['temperature'].iloc[i] >= temp_p90):
                        whiplash_events.append({
                            'date': climate_data.index[i],
                            'type': 'cold_to_hot',
                            'temperature_change': climate_data['temperature'].iloc[i] - climate_data['temperature'].iloc[i-1],
                            'severity': 'extreme'
                        })
        
        return whiplash_events
    
    def analyze_climate_trend_acceleration(self, climate_data):
        """Analyze acceleration of climate trends using polynomial fitting"""
        trend_analysis = {}
        
        if isinstance(climate_data, pd.DataFrame):
            for col in climate_data.select_dtypes(include=[np.number]).columns:
                data = climate_data[col].values
                x = np.arange(len(data))
                
                # Linear trend
                linear_coeffs = np.polyfit(x, data, 1)
                linear_trend = linear_coeffs[0]
                
                # Quadratic trend (acceleration)
                quad_coeffs = np.polyfit(x, data, 2)
                acceleration = 2 * quad_coeffs[0]  # Second derivative
                
                # Cubic trend (jerk/change in acceleration)
                cubic_coeffs = np.polyfit(x, data, 3)
                jerk = 6 * cubic_coeffs[0]  # Third derivative
                
                trend_analysis[col] = {
                    'linear_trend': linear_trend,
                    'acceleration': acceleration,
                    'jerk': jerk,
                    'trend_direction': 'increasing' if linear_trend > 0 else 'decreasing',
                    'acceleration_direction': 'accelerating' if acceleration > 0 else 'decelerating',
                    'complexity': 'high' if abs(jerk) > abs(acceleration) * 0.1 else 'moderate'
                }
        
        return trend_analysis
    
    def generate_climate_scenarios(self, base_data, scenario_types=['ssp1', 'ssp3', 'ssp5']):
        """Generate climate scenarios using GAN or statistical methods"""
        scenarios = {}
        
        if isinstance(base_data, pd.DataFrame):
            # Normalize data
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(base_data.select_dtypes(include=[np.number]))
            
            # Build and train GAN for scenario generation
            latent_dim = 100
            climate_dim = normalized_data.shape[1]
            
            generator, discriminator, gan = self.build_climate_gan(latent_dim, climate_dim)
            
            # Train GAN (simplified training)
            epochs = 1000
            batch_size = 32
            
            for epoch in range(epochs):
                # Train discriminator
                idx = np.random.randint(0, normalized_data.shape[0], batch_size)
                real_climate = normalized_data[idx]
                
                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                fake_climate = generator.predict(noise, verbose=0)
                
                d_loss_real = discriminator.train_on_batch(real_climate, np.ones((batch_size, 1)))
                d_loss_fake = discriminator.train_on_batch(fake_climate, np.zeros((batch_size, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # Train generator
                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
                
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}: D_loss={d_loss[0]}, G_loss={g_loss}")
            
            # Generate scenarios
            for scenario in scenario_types:
                # Different noise for different scenarios
                if scenario == 'ssp1':  # Sustainability
                    noise_scale = 0.5
                elif scenario == 'ssp3':  # Regional rivalry
                    noise_scale = 1.5
                elif scenario == 'ssp5':  # Fossil-fueled development
                    noise_scale = 2.0
                else:
                    noise_scale = 1.0
                
                noise = np.random.normal(0, noise_scale, (len(base_data), latent_dim))
                generated_scenario = generator.predict(noise, verbose=0)
                
                # Denormalize
                scenario_data = scaler.inverse_transform(generated_scenario)
                scenario_df = pd.DataFrame(
                    scenario_data, 
                    columns=base_data.select_dtypes(include=[np.number]).columns,
                    index=base_data.index
                )
                
                scenarios[scenario] = scenario_df
        
        return scenarios
