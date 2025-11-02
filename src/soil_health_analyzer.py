"""
Advanced Soil Health Analysis using Deep Learning and Neural Networks
State-of-the-art models for soil classification, health assessment, and prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SoilHealthAnalyzer:
    """
    Advanced soil health analysis using ensemble of deep learning models
    and traditional ML algorithms for comprehensive soil assessment
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.soil_health_index = None
        
    def build_soil_health_cnn(self, input_shape):
        """Build Convolutional Neural Network for spatial soil data analysis"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(1, activation='sigmoid')  # Soil health score
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def build_soil_lstm(self, timesteps, features):
        """Build LSTM model for temporal soil health analysis"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.LSTM(64, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.LSTM(32, return_sequences=False),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def build_soil_transformer(self, input_shape):
        """Build Transformer model for soil health prediction"""
        inputs = layers.Input(shape=input_shape)
        
        # Positional encoding
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        position_embeddings = layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(positions)
        
        # Add positional encoding to inputs
        x = inputs + position_embeddings
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=input_shape[1]
        )(x, x)
        
        # Add & Norm
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
        
        # Feed forward
        ff_output = layers.Dense(input_shape[1] * 4, activation='relu')(x)
        ff_output = layers.Dense(input_shape[1])(ff_output)
        
        # Add & Norm
        x = layers.Add()([x, ff_output])
        x = layers.LayerNormalization()(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def build_ensemble_model(self, X_train, y_train):
        """Build ensemble of advanced models for robust predictions"""
        models_dict = {}
        
        # Deep Learning Models
        if len(X_train.shape) == 3:  # Sequential data
            models_dict['lstm'] = self.build_soil_lstm(X_train.shape[1], X_train.shape[2])
        elif len(X_train.shape) == 4:  # Image data
            models_dict['cnn'] = self.build_soil_health_cnn(X_train.shape[1:])
        else:  # Tabular data
            models_dict['transformer'] = self.build_soil_transformer((X_train.shape[1], 1))
        
        # Gradient Boosting Models
        models_dict['xgboost'] = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        models_dict['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        models_dict['random_forest'] = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Train models
        for name, model in models_dict.items():
            print(f"Training {name} model...")
            
            if name in ['lstm', 'cnn', 'transformer']:
                # Deep learning models
                early_stopping = callbacks.EarlyStopping(
                    monitor='val_loss', patience=20, restore_best_weights=True
                )
                reduce_lr = callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6
                )
                
                if name == 'transformer':
                    X_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                    model.fit(
                        X_reshaped, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping, reduce_lr],
                        verbose=0
                    )
                else:
                    model.fit(
                        X_train, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping, reduce_lr],
                        verbose=0
                    )
            else:
                # Traditional ML models
                model.fit(X_train, y_train)
            
            models_dict[name] = model
        
        self.models = models_dict
        return models_dict
    
    def calculate_soil_health_index(self, soil_data):
        """
        Calculate comprehensive soil health index using multiple indicators
        """
        if isinstance(soil_data, pd.DataFrame):
            health_components = {}
            
            # pH component (optimal range 6.0-7.0)
            if 'ph' in soil_data.columns:
                ph_score = 1 - np.abs(soil_data['ph'] - 6.5) / 3.5  # Normalize to 0-1
                health_components['ph_score'] = np.clip(ph_score, 0, 1)
            
            # Organic carbon component
            if 'organic_carbon' in soil_data.columns:
                oc_score = np.tanh(soil_data['organic_carbon'] / 2)  # Saturates around 2%
                health_components['organic_carbon_score'] = np.clip(oc_score, 0, 1)
            
            # Texture component (balanced loam is ideal)
            if 'sand_content' in soil_data.columns and 'clay_content' in soil_data.columns:
                # Ideal loam: ~40% sand, 40% silt, 20% clay
                silt_content = 100 - soil_data['sand_content'] - soil_data['clay_content']
                texture_score = 1 - (
                    np.abs(soil_data['sand_content'] - 40) / 40 +
                    np.abs(silt_content - 40) / 40 +
                    np.abs(soil_data['clay_content'] - 20) / 20
                ) / 3
                health_components['texture_score'] = np.clip(texture_score, 0, 1)
            
            # Erosion risk component (inverse)
            if 'erosion_risk' in soil_data.columns:
                erosion_score = 1 - np.clip(soil_data['erosion_risk'] / 100, 0, 1)
                health_components['erosion_score'] = erosion_score
            
            # Nutrient components
            nutrient_cols = ['nitrogen', 'phosphorus', 'potassium']
            for nutrient in nutrient_cols:
                if nutrient in soil_data.columns:
                    # Z-score normalization assuming normal distribution
                    nutrient_score = stats.norm.cdf(
                        (soil_data[nutrient] - soil_data[nutrient].mean()) / 
                        soil_data[nutrient].std()
                    )
                    health_components[f'{nutrient}_score'] = nutrient_score
            
            # Calculate weighted soil health index
            if health_components:
                weights = {
                    'ph_score': 0.2,
                    'organic_carbon_score': 0.25,
                    'texture_score': 0.15,
                    'erosion_score': 0.2,
                    'nitrogen_score': 0.1,
                    'phosphorus_score': 0.05,
                    'potassium_score': 0.05
                }
                
                # Use available components with their weights
                total_weight = sum(weights.get(comp, 0.1) for comp in health_components.keys())
                weighted_score = sum(
                    health_components[comp] * weights.get(comp, 0.1) 
                    for comp in health_components.keys()
                ) / total_weight
                
                self.soil_health_index = weighted_score
                return weighted_score
        
        return None
    
    def predict_soil_health(self, X):
        """Make ensemble predictions for soil health"""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if name in ['lstm', 'cnn', 'transformer']:
                    if name == 'transformer':
                        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
                        pred = model.predict(X_reshaped, verbose=0)
                    else:
                        pred = model.predict(X, verbose=0)
                else:
                    pred = model.predict(X)
                
                predictions[name] = pred.flatten()
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                continue
        
        # Ensemble predictions (weighted average)
        if predictions:
            # Give more weight to deep learning models
            weights = {
                'lstm': 0.3, 'cnn': 0.3, 'transformer': 0.25,
                'xgboost': 0.1, 'lightgbm': 0.03, 'random_forest': 0.02
            }
            
            ensemble_pred = np.zeros(len(list(predictions.values())[0]))
            total_weight = 0
            
            for name, pred in predictions.items():
                weight = weights.get(name, 0.1)
                ensemble_pred += pred * weight
                total_weight += weight
            
            ensemble_pred /= total_weight
            predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def analyze_soil_degradation_trends(self, temporal_data):
        """Analyze soil degradation trends using time series analysis"""
        if isinstance(temporal_data, pd.DataFrame):
            degradation_trends = {}
            
            # Calculate trend for each soil property
            numeric_cols = temporal_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col in temporal_data.columns:
                    # Linear trend
                    x = np.arange(len(temporal_data))
                    y = temporal_data[col].values
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    degradation_trends[col] = {
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'trend_direction': 'improving' if slope > 0 else 'degrading',
                        'significance': 'significant' if p_value < 0.05 else 'not_significant'
                    }
            
            return degradation_trends
        
        return None
    
    def identify_soil_health_hotspots(self, gdf, health_score_col='health_index'):
        """Identify soil health hotspots using spatial clustering"""
        if isinstance(gdf, pd.DataFrame) and health_score_col in gdf.columns:
            from sklearn.cluster import DBSCAN
            from scipy.spatial.distance import cdist
            
            # Extract coordinates and health scores
            coords = np.array([[geom.x, geom.y] for geom in gdf.geometry.centroid])
            health_scores = gdf[health_score_col].values.reshape(-1, 1)
            
            # Combine spatial and health features
            features = np.hstack([coords, health_scores])
            
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # DBSCAN clustering
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(features_scaled)
            gdf['cluster'] = clustering.labels_
            
            # Identify hotspots (clusters with low health scores)
            hotspots = []
            for cluster_id in set(clustering.labels_):
                if cluster_id != -1:  # Ignore noise
                    cluster_mask = gdf['cluster'] == cluster_id
                    cluster_health = gdf.loc[cluster_mask, health_score_col].mean()
                    
                    if cluster_health < 0.3:  # Low health threshold
                        hotspots.append({
                            'cluster_id': cluster_id,
                            'mean_health_score': cluster_health,
                            'area_size': cluster_mask.sum(),
                            'centroid': gdf.loc[cluster_mask, 'geometry'].unary_union.centroid
                        })
            
            return hotspots
        
        return None
    
    def generate_soil_health_recommendations(self, soil_data, health_scores):
        """Generate personalized soil health recommendations"""
        recommendations = []
        
        if isinstance(soil_data, pd.DataFrame) and len(health_scores) == len(soil_data):
            for i, (_, row) in enumerate(soil_data.iterrows()):
                score = health_scores[i]
                rec = {"location": i, "health_score": score, "recommendations": []}
                
                if score < 0.3:
                    rec["urgency"] = "Critical"
                    
                    if 'ph' in row and row['ph'] < 5.5:
                        rec["recommendations"].append("Apply agricultural lime to raise pH")
                    elif 'ph' in row and row['ph'] > 7.5:
                        rec["recommendations"].append("Apply sulfur or acidic amendments to lower pH")
                    
                    if 'organic_carbon' in row and row['organic_carbon'] < 1:
                        rec["recommendations"].append("Add organic matter: compost, manure, or cover crops")
                    
                    if 'erosion_risk' in row and row['erosion_risk'] > 50:
                        rec["recommendations"].append("Implement erosion control: contour plowing, terracing")
                
                elif score < 0.6:
                    rec["urgency"] = "Moderate"
                    rec["recommendations"].append("Monitor soil health regularly")
                    rec["recommendations"].append("Consider conservation agriculture practices")
                
                else:
                    rec["urgency"] = "Good"
                    rec["recommendations"].append("Maintain current soil management practices")
                    rec["recommendations"].append("Implement precision agriculture for optimization")
                
                recommendations.append(rec)
        
        return recommendations
