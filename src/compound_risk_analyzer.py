"""
Advanced Compound Risk Analysis using Deep Learning and Ensemble Methods
State-of-the-art models for analyzing combined soil and climate risks
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class CompoundRiskAnalyzer:
    """
    Advanced compound risk analysis using deep learning and ensemble methods
    for identifying locations with overlapping soil and climate stress
    """
    
    def __init__(self):
        self.risk_models = {}
        self.clustering_models = {}
        self.risk_maps = {}
        self.compound_indices = {}
        
    def build_compound_risk_neural_network(self, input_dim):
        """Build deep neural network for compound risk assessment"""
        inputs = layers.Input(shape=(input_dim,))
        
        # Feature extraction branches
        # Soil branch
        soil_branch = layers.Dense(64, activation='relu')(inputs[:, :input_dim//2])
        soil_branch = layers.BatchNormalization()(soil_branch)
        soil_branch = layers.Dropout(0.3)(soil_branch)
        soil_branch = layers.Dense(32, activation='relu')(soil_branch)
        soil_branch = layers.BatchNormalization()(soil_branch)
        
        # Climate branch
        climate_branch = layers.Dense(64, activation='relu')(inputs[:, input_dim//2:])
        climate_branch = layers.BatchNormalization()(climate_branch)
        climate_branch = layers.Dropout(0.3)(climate_branch)
        climate_branch = layers.Dense(32, activation='relu')(climate_branch)
        climate_branch = layers.BatchNormalization()(climate_branch)
        
        # Interaction layer
        interaction = layers.Concatenate()([soil_branch, climate_branch])
        interaction = layers.Dense(64, activation='relu')(interaction)
        interaction = layers.BatchNormalization()(interaction)
        interaction = layers.Dropout(0.3)(interaction)
        
        # Attention mechanism
        attention = layers.Dense(32, activation='tanh')(interaction)
        attention = layers.Dense(32, activation='softmax')(attention)
        attended = layers.Multiply()([interaction, attention])
        
        # Risk classification layers
        risk_features = layers.Dense(128, activation='relu')(attended)
        risk_features = layers.BatchNormalization()(risk_features)
        risk_features = layers.Dropout(0.4)(risk_features)
        
        risk_features = layers.Dense(64, activation='relu')(risk_features)
        risk_features = layers.BatchNormalization()(risk_features)
        risk_features = layers.Dropout(0.3)(risk_features)
        
        # Multi-task outputs
        compound_risk = layers.Dense(1, activation='sigmoid', name='compound_risk')(risk_features)
        soil_risk = layers.Dense(1, activation='sigmoid', name='soil_risk')(risk_features)
        climate_risk = layers.Dense(1, activation='sigmoid', name='climate_risk')(risk_features)
        severity = layers.Dense(3, activation='softmax', name='severity')(risk_features)  # Low, Medium, High
        
        model = keras.Model(
            inputs=inputs,
            outputs=[compound_risk, soil_risk, climate_risk, severity]
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'compound_risk': 'binary_crossentropy',
                'soil_risk': 'binary_crossentropy',
                'climate_risk': 'binary_crossentropy',
                'severity': 'categorical_crossentropy'
            },
            loss_weights={
                'compound_risk': 0.4,
                'soil_risk': 0.2,
                'climate_risk': 0.2,
                'severity': 0.2
            },
            metrics={
                'compound_risk': ['accuracy', 'auc'],
                'soil_risk': ['accuracy'],
                'climate_risk': ['accuracy'],
                'severity': ['accuracy']
            }
        )
        
        return model
    
    def build_risk_gan(self, latent_dim, risk_dim):
        """Build GAN for generating realistic compound risk scenarios"""
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
        
        generator_output = layers.Dense(risk_dim, activation='sigmoid')(gen)
        
        generator = keras.Model(generator_input, generator_output)
        
        # Discriminator
        discriminator_input = layers.Input(shape=(risk_dim,))
        
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
    
    def calculate_compound_risk_indices(self, soil_data, climate_data):
        """Calculate comprehensive compound risk indices"""
        compound_indices = {}
        
        if isinstance(soil_data, pd.DataFrame) and isinstance(climate_data, pd.DataFrame):
            # Ensure data alignment
            common_index = soil_data.index.intersection(climate_data.index)
            soil_aligned = soil_data.loc[common_index]
            climate_aligned = climate_data.loc[common_index]
            
            # Soil risk components
            soil_risk_components = {}
            if 'ph' in soil_aligned.columns:
                ph_risk = 1 - (1 - np.abs(soil_aligned['ph'] - 6.5) / 3.5)  # Inverse of pH score
                soil_risk_components['ph_risk'] = np.clip(ph_risk, 0, 1)
            
            if 'organic_carbon' in soil_aligned.columns:
                carbon_risk = 1 - np.tanh(soil_aligned['organic_carbon'] / 2)
                soil_risk_components['carbon_risk'] = np.clip(carbon_risk, 0, 1)
            
            if 'erosion_risk' in soil_aligned.columns:
                erosion_risk = soil_aligned['erosion_risk'] / 100
                soil_risk_components['erosion_risk'] = np.clip(erosion_risk, 0, 1)
            
            # Climate risk components
            climate_risk_components = {}
            if 'temperature_stress' in climate_aligned.columns:
                climate_risk_components['temperature_risk'] = climate_aligned['temperature_stress']
            
            if 'precipitation_stress' in climate_aligned.columns:
                climate_risk_components['precipitation_risk'] = climate_aligned['precipitation_stress']
            
            if 'drought_index' in climate_aligned.columns:
                drought_risk = np.clip(-climate_aligned['drought_index'], 0, 1)
                climate_risk_components['drought_risk'] = drought_risk
            
            # Compound risk calculations
            if soil_risk_components and climate_risk_components:
                # Weighted soil risk
                soil_weights = {'ph_risk': 0.3, 'carbon_risk': 0.4, 'erosion_risk': 0.3}
                soil_risk = sum(
                    soil_risk_components.get(comp, 0) * soil_weights.get(comp, 0.1)
                    for comp in soil_risk_components.keys()
                ) / sum(soil_weights.get(comp, 0.1) for comp in soil_risk_components.keys())
                
                # Weighted climate risk
                climate_weights = {'temperature_risk': 0.3, 'precipitation_risk': 0.3, 'drought_risk': 0.4}
                climate_risk = sum(
                    climate_risk_components.get(comp, 0) * climate_weights.get(comp, 0.1)
                    for comp in climate_risk_components.keys()
                ) / sum(climate_weights.get(comp, 0.1) for comp in climate_risk_components.keys())
                
                # Compound risk indices
                compound_indices['soil_risk_index'] = soil_risk
                compound_indices['climate_risk_index'] = climate_risk
                
                # Multiplicative compound risk (high when both are high)
                compound_indices['compound_risk_multiplicative'] = soil_risk * climate_risk
                
                # Additive compound risk with interaction term
                compound_indices['compound_risk_additive'] = (soil_risk + climate_risk) / 2
                
                # Maximum risk (conservative approach)
                compound_indices['compound_risk_maximum'] = np.maximum(soil_risk, climate_risk)
                
                # Risk synergy (interaction effect)
                compound_indices['risk_synergy'] = soil_risk * climate_risk * 2  # Amplifies when both are high
                
                # Risk categories
                compound_risk = compound_indices['compound_risk_multiplicative']
                compound_indices['risk_category'] = pd.cut(
                    compound_risk,
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=['Low', 'Medium', 'High']
                )
                
                # Risk urgency (time-sensitive component)
                if 'temperature_trend' in climate_aligned.columns:
                    trend_factor = np.clip(climate_aligned['temperature_trend'] * 10, 0, 1)
                    compound_indices['risk_urgency'] = compound_risk * (1 + trend_factor)
        
        self.compound_indices = compound_indices
        return compound_indices
    
    def identify_risk_hotspots(self, gdf, risk_col='compound_risk_multiplicative'):
        """Identify compound risk hotspots using advanced clustering"""
        if isinstance(gdf, pd.DataFrame) and risk_col in gdf.columns:
            hotspots = {}
            
            # Extract spatial coordinates and risk values
            coords = np.array([[geom.x, geom.y] for geom in gdf.geometry.centroid])
            risk_values = gdf[risk_col].values.reshape(-1, 1)
            
            # Combine spatial and risk features
            features = np.hstack([coords, risk_values])
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Multiple clustering approaches
            clustering_methods = {
                'dbscan': DBSCAN(eps=0.5, min_samples=5),
                'kmeans': KMeans(n_clusters=5, random_state=42),
                'gaussian_mixture': GaussianMixture(n_components=5, random_state=42)
            }
            
            for method_name, model in clustering_methods.items():
                if method_name == 'gaussian_mixture':
                    cluster_labels = model.fit_predict(features_scaled)
                else:
                    cluster_labels = model.fit_predict(features_scaled)
                
                gdf[f'cluster_{method_name}'] = cluster_labels
                
                # Analyze clusters for hotspots
                method_hotspots = []
                for cluster_id in set(cluster_labels):
                    if cluster_id != -1:  # Ignore noise for DBSCAN
                        cluster_mask = gdf[f'cluster_{method_name}'] == cluster_id
                        cluster_risk = gdf.loc[cluster_mask, risk_col].mean()
                        cluster_size = cluster_mask.sum()
                        
                        if cluster_risk > 0.7 and cluster_size >= 3:  # High risk threshold
                            cluster_geom = gdf.loc[cluster_mask, 'geometry'].unary_union.centroid
                            
                            method_hotspots.append({
                                'cluster_id': cluster_id,
                                'mean_risk': cluster_risk,
                                'size': cluster_size,
                                'centroid': (cluster_geom.x, cluster_geom.y),
                                'risk_level': 'Critical' if cluster_risk > 0.85 else 'High'
                            })
                
                hotspots[method_name] = method_hotspots
            
            # Consensus hotspots (identified by multiple methods)
            all_hotspots = []
            for method_spots in hotspots.values():
                all_hotspots.extend([(spot['centroid'][0], spot['centroid'][1]) for spot in method_spots])
            
            if all_hotspots:
                # Find consensus locations
                consensus_hotspots = []
                for i, spot1 in enumerate(all_hotspots):
                    nearby_count = sum(1 for spot2 in all_hotspots 
                                     if np.sqrt((spot1[0]-spot2[0])**2 + (spot1[1]-spot2[1])**2) < 0.5)
                    if nearby_count >= 2:  # Identified by at least 2 methods
                        consensus_hotspots.append(spot1)
                
                hotspots['consensus'] = consensus_hotspots
        
        return hotspots
    
    def analyze_agricultural_exposure(self, risk_data, agricultural_data):
        """Analyze agricultural exposure to compound risks"""
        exposure_analysis = {}
        
        if isinstance(risk_data, pd.DataFrame) and isinstance(agricultural_data, pd.DataFrame):
            # Align data
            common_index = risk_data.index.intersection(agricultural_data.index)
            risk_aligned = risk_data.loc[common_index]
            agri_aligned = agricultural_data.loc[common_index]
            
            # Calculate exposure metrics
            exposure_metrics = {}
            
            # Crop exposure
            crop_cols = [col for col in agri_aligned.columns if 'crop' in col.lower()]
            for crop_col in crop_cols:
                if crop_col in agri_aligned.columns:
                    crop_area = agri_aligned[crop_col]
                    high_risk_mask = risk_aligned['compound_risk_multiplicative'] > 0.7
                    
                    exposed_area = (crop_area * high_risk_mask).sum()
                    total_area = crop_area.sum()
                    exposure_percentage = (exposed_area / total_area * 100) if total_area > 0 else 0
                    
                    exposure_metrics[f'{crop_col}_exposure'] = {
                        'exposed_area': exposed_area,
                        'total_area': total_area,
                        'exposure_percentage': exposure_percentage,
                        'risk_level': 'Critical' if exposure_percentage > 50 else 'High' if exposure_percentage > 25 else 'Medium'
                    }
            
            # Livestock exposure
            livestock_cols = [col for col in agri_aligned.columns if 'livestock' in col.lower() or 'animal' in col.lower()]
            for livestock_col in livestock_cols:
                if livestock_col in agri_aligned.columns:
                    livestock_count = agri_aligned[livestock_count]
                    high_risk_mask = risk_aligned['compound_risk_multiplicative'] > 0.7
                    
                    exposed_livestock = (livestock_count * high_risk_mask).sum()
                    total_livestock = livestock_count.sum()
                    exposure_percentage = (exposed_livestock / total_livestock * 100) if total_livestock > 0 else 0
                    
                    exposure_metrics[f'{livestock_col}_exposure'] = {
                        'exposed_count': exposed_livestock,
                        'total_count': total_livestock,
                        'exposure_percentage': exposure_percentage,
                        'risk_level': 'Critical' if exposure_percentage > 50 else 'High' if exposure_percentage > 25 else 'Medium'
                    }
            
            # Economic exposure (if economic data available)
            if 'economic_value' in agri_aligned.columns:
                economic_value = agri_aligned['economic_value']
                high_risk_mask = risk_aligned['compound_risk_multiplicative'] > 0.7
                
                exposed_value = (economic_value * high_risk_mask).sum()
                total_value = economic_value.sum()
                value_exposure_percentage = (exposed_value / total_value * 100) if total_value > 0 else 0
                
                exposure_metrics['economic_exposure'] = {
                    'exposed_value': exposed_value,
                    'total_value': total_value,
                    'exposure_percentage': value_exposure_percentage,
                    'potential_loss': exposed_value * 0.5,  # Assume 50% loss under high risk
                    'risk_level': 'Critical' if value_exposure_percentage > 50 else 'High' if value_exposure_percentage > 25 else 'Medium'
                }
            
            exposure_analysis['exposure_metrics'] = exposure_metrics
            
            # Vulnerability index (combining exposure and sensitivity)
            if exposure_metrics:
                vulnerability_scores = []
                for metric in exposure_metrics.values():
                    if 'exposure_percentage' in metric:
                        vulnerability_scores.append(metric['exposure_percentage'] / 100)
                
                if vulnerability_scores:
                    overall_vulnerability = np.mean(vulnerability_scores)
                    exposure_analysis['overall_vulnerability_index'] = overall_vulnerability
                    exposure_analysis['vulnerability_level'] = (
                        'Extreme' if overall_vulnerability > 0.8 else
                        'High' if overall_vulnerability > 0.6 else
                        'Medium' if overall_vulnerability > 0.4 else
                        'Low'
                    )
        
        return exposure_analysis
    
    def predict_compound_risk_evolution(self, historical_risk_data, years_ahead=10):
        """Predict evolution of compound risk using time series models"""
        evolution_predictions = {}
        
        if isinstance(historical_risk_data, pd.DataFrame):
            risk_components = ['compound_risk_multiplicative', 'soil_risk_index', 'climate_risk_index']
            
            for component in risk_components:
                if component in historical_risk_data.columns:
                    data = historical_risk_data[component].values
                    
                    # Prepare sequences for LSTM
                    sequence_length = 12  # Monthly data
                    X, y = [], []
                    
                    for i in range(len(data) - sequence_length):
                        X.append(data[i:i+sequence_length])
                        y.append(data[i+sequence_length])
                    
                    X = np.array(X)
                    y = np.array(y)
                    
                    if len(X) > 0:
                        # Build LSTM model
                        model = models.Sequential([
                            layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
                            layers.Dropout(0.2),
                            layers.LSTM(50, activation='relu', return_sequences=False),
                            layers.Dropout(0.2),
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
                        
                        for _ in range(years_ahead * 12):
                            next_val = model.predict(future_seq, verbose=0)[0, 0]
                            future_vals.append(np.clip(next_val, 0, 1))
                            
                            # Update sequence
                            future_seq = np.roll(future_seq, -1, axis=1)
                            future_seq[0, -1, 0] = next_val
                        
                        evolution_predictions[component] = future_vals
                        
                        # Calculate trend acceleration
                        if len(future_vals) > 2:
                            trend_slope = np.polyfit(range(len(future_vals)), future_vals, 1)[0]
                            trend_acceleration = np.polyfit(range(len(future_vals)), future_vals, 2)[0] * 2
                            
                            evolution_predictions[f'{component}_trend_slope'] = trend_slope
                            evolution_predictions[f'{component}_trend_acceleration'] = trend_acceleration
            
            # Risk category evolution
            if 'compound_risk_multiplicative' in evolution_predictions:
                future_risk = evolution_predictions['compound_risk_multiplicative']
                risk_categories = []
                
                for risk_val in future_risk:
                    if risk_val > 0.7:
                        risk_categories.append('High')
                    elif risk_val > 0.4:
                        risk_categories.append('Medium')
                    else:
                        risk_categories.append('Low')
                
                evolution_predictions['risk_category_evolution'] = risk_categories
                
                # Calculate probability of crossing risk thresholds
                high_risk_probability = np.mean(np.array(future_risk) > 0.7)
                medium_risk_probability = np.mean((np.array(future_risk) > 0.4) & (np.array(future_risk) <= 0.7))
                
                evolution_predictions['high_risk_probability'] = high_risk_probability
                evolution_predictions['medium_risk_probability'] = medium_risk_probability
        
        return evolution_predictions
    
    def generate_adaptation_priorities(self, risk_data, agricultural_data, budget_constraint=None):
        """Generate data-driven adaptation priorities using optimization"""
        adaptation_priorities = {}
        
        if isinstance(risk_data, pd.DataFrame) and isinstance(agricultural_data, pd.DataFrame):
            # Calculate priority scores
            priority_factors = {}
            
            # Risk severity
            priority_factors['risk_severity'] = risk_data['compound_risk_multiplicative']
            
            # Agricultural importance
            if 'economic_value' in agricultural_data.columns:
                priority_factors['economic_importance'] = agricultural_data['economic_value'] / agricultural_data['economic_value'].sum()
            else:
                priority_factors['economic_importance'] = 1.0
            
            # Population exposure (if available)
            if 'population' in agricultural_data.columns:
                priority_factors['population_exposure'] = agricultural_data['population'] / agricultural_data['population'].sum()
            else:
                priority_factors['population_exposure'] = 1.0
            
            # Adaptation feasibility (inverse of implementation difficulty)
            # This would typically come from expert knowledge - using placeholder
            priority_factors['adaptation_feasibility'] = 0.7  # Assume 70% feasibility on average
            
            # Calculate composite priority score
            priority_score = (
                priority_factors['risk_severity'] * 0.4 +
                priority_factors['economic_importance'] * 0.3 +
                priority_factors['population_exposure'] * 0.2 +
                priority_factors['adaptation_feasibility'] * 0.1
            )
            
            adaptation_priorities['priority_scores'] = priority_score
            
            # Rank locations by priority
            priority_ranking = priority_score.sort_values(ascending=False)
            adaptation_priorities['priority_ranking'] = priority_ranking
            
            # Generate adaptation recommendations based on dominant risks
            recommendations = []
            
            for idx in priority_ranking.head(20).index:  # Top 20 priorities
                location_risks = risk_data.loc[idx]
                location_agri = agricultural_data.loc[idx]
                
                rec = {
                    'location_id': idx,
                    'priority_score': priority_score.loc[idx],
                    'dominant_risks': [],
                    'recommended_actions': [],
                    'estimated_cost': 0,
                    'expected_benefit': 0
                }
                
                # Identify dominant risks
                if location_risks.get('soil_risk_index', 0) > 0.6:
                    rec['dominant_risks'].append('soil_degradation')
                    rec['recommended_actions'].extend([
                        'soil_amendment_application',
                        'conservation_agriculture',
                        'cover_cropping'
                    ])
                    rec['estimated_cost'] += 100000  # Placeholder cost
                
                if location_risks.get('climate_risk_index', 0) > 0.6:
                    rec['dominant_risks'].append('climate_stress')
                    rec['recommended_actions'].extend([
                        'drought_resistant_varieties',
                        'irrigation_infrastructure',
                        'water_harvesting_systems'
                    ])
                    rec['estimated_cost'] += 150000  # Placeholder cost
                
                # Calculate expected benefit (simplified)
                rec['expected_benefit'] = rec['priority_score'] * location_agri.get('economic_value', 1000000) * 0.3
                
                recommendations.append(rec)
            
            adaptation_priorities['recommendations'] = recommendations
            
            # Budget optimization (if budget constraint provided)
            if budget_constraint:
                # Simple knapsack optimization for budget allocation
                from itertools import combinations
                
                best_allocation = []
                remaining_budget = budget_constraint
                
                # Sort by benefit-cost ratio
                recommendations.sort(key=lambda x: x['expected_benefit'] / max(x['estimated_cost'], 1), reverse=True)
                
                for rec in recommendations:
                    if rec['estimated_cost'] <= remaining_budget:
                        best_allocation.append(rec)
                        remaining_budget -= rec['estimated_cost']
                
                adaptation_priorities['budget_optimized_allocation'] = best_allocation
                adaptation_priorities['remaining_budget'] = remaining_budget
                adaptation_priorities['total_expected_benefit'] = sum(r['expected_benefit'] for r in best_allocation)
        
        return adaptation_priorities
