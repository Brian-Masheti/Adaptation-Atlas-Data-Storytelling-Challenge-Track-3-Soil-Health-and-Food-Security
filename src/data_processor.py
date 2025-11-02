"""
Advanced Data Processing Pipeline for Soil Health and Climate Analysis
Utilizing TensorFlow and cutting-edge ML techniques for data preprocessing
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
import warnings
warnings.filterwarnings('ignore')

class AdvancedDataProcessor:
    """
    State-of-the-art data processing pipeline with deep learning integration
    for soil health and climate data analysis
    """
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.dimensionality_reducers = {}
        self.feature_extractors = {}
        
    def load_climate_data(self, file_paths):
        """Load and preprocess climate data from multiple sources"""
        climate_datasets = []
        
        for path in file_paths:
            if path.endswith('.nc'):
                ds = xr.open_dataset(path)
                climate_datasets.append(ds)
            elif path.endswith('.csv'):
                df = pd.read_csv(path)
                climate_datasets.append(df)
        
        return climate_datasets
    
    def load_soil_data(self, soilgrids_path, additional_paths=None):
        """Load soil data from SoilGrids and other sources"""
        soil_data = {}
        
        # Load SoilGrids data
        if soilgrids_path.endswith('.nc'):
            soil_data['soilgrids'] = xr.open_dataset(soilgrids_path)
        
        # Load additional soil data
        if additional_paths:
            for path in additional_paths:
                if path.endswith('.shp'):
                    gdf = gpd.read_file(path)
                    soil_data[f'soil_{path.split("/")[-1]}'] = gdf
                elif path.endswith('.csv'):
                    df = pd.read_csv(path)
                    soil_data[f'soil_{path.split("/")[-1]}'] = df
        
        return soil_data
    
    def advanced_imputation(self, data, method='knn', n_neighbors=5):
        """Advanced missing data imputation using multiple strategies"""
        if isinstance(data, pd.DataFrame):
            if method == 'knn':
                imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
                imputed = imputer.fit_transform(data)
                self.imputers['knn'] = imputer
                return pd.DataFrame(imputed, columns=data.columns, index=data.index)
            
            elif method == 'iterative':
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                imputer = IterativeImputer(random_state=42, max_iter=10)
                imputed = imputer.fit_transform(data)
                self.imputers['iterative'] = imputer
                return pd.DataFrame(imputed, columns=data.columns, index=data.index)
            
            elif method == 'deep_learning':
                # Deep learning-based imputation using autoencoders
                return self._deep_learning_imputation(data)
        
        return data
    
    def _deep_learning_imputation(self, data):
        """Deep learning-based imputation using variational autoencoders"""
        # Create mask for missing values
        mask = data.notna().astype(int)
        
        # Simple imputation for initial fill
        simple_imputer = SimpleImputer(strategy='median')
        data_filled = simple_imputer.fit_transform(data)
        
        # Build variational autoencoder for imputation
        input_dim = data.shape[1]
        
        # Encoder
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        encoded = tf.keras.layers.Dense(128, activation='relu')(inputs)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)
        encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
        
        # Latent space
        z_mean = tf.keras.layers.Dense(32)(encoded)
        z_log_var = tf.keras.layers.Dense(32)(encoded)
        
        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon
        
        z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
        
        # Decoder
        decoded = tf.keras.layers.Dense(64, activation='relu')(z)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
        outputs = tf.keras.layers.Dense(input_dim, activation='linear')(decoded)
        
        # Build and compile model
        vae = tf.keras.Model(inputs, outputs)
        vae.compile(optimizer='adam', loss='mse')
        
        # Train the model
        vae.fit(data_filled, data_filled, epochs=100, batch_size=32, verbose=0)
        
        # Generate improved imputation
        improved_data = vae.predict(data_filled)
        
        return pd.DataFrame(improved_data, columns=data.columns, index=data.index)
    
    def feature_engineering(self, data):
        """Advanced feature engineering with domain expertise"""
        if isinstance(data, pd.DataFrame):
            engineered_data = data.copy()
            
            # Soil health indices
            if 'ph' in engineered_data.columns:
                engineered_data['ph_category'] = pd.cut(engineered_data['ph'], 
                                                       bins=[0, 5.5, 6.5, 7.5, 14], 
                                                       labels=['Very Acidic', 'Acidic', 'Neutral', 'Alkaline'])
            
            if 'organic_carbon' in engineered_data.columns and 'clay_content' in engineered_data.columns:
                engineered_data['carbon_sequestration_potential'] = (
                    engineered_data['organic_carbon'] * engineered_data['clay_content'] / 100
                )
            
            # Climate stress indices
            if 'temperature' in engineered_data.columns and 'precipitation' in engineered_data.columns:
                engineered_data['aridity_index'] = (
                    engineered_data['precipitation'] / (engineered_data['temperature'] + 273.15)
                )
                
                # Growing degree days
                engineered_data['growing_degree_days'] = np.maximum(0, 
                    engineered_data['temperature'] - 10)  # Base temperature 10°C
            
            # Compound risk indicators
            if 'erosion_risk' in engineered_data.columns and 'moisture_stress' in engineered_data.columns:
                engineered_data['compound_risk_index'] = (
                    engineered_data['erosion_risk'] * engineered_data['moisture_stress']
                )
            
            return engineered_data
        
        return data
    
    def advanced_scaling(self, data, method='robust'):
        """Advanced scaling techniques for different data distributions"""
        if isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if method == 'robust':
                scaler = RobustScaler()
                data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
                self.scalers['robust'] = scaler
                
            elif method == 'standard':
                scaler = StandardScaler()
                data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
                self.scalers['standard'] = scaler
                
            elif method == 'minmax':
                scaler = MinMaxScaler()
                data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
                self.scalers['minmax'] = scaler
                
            elif method == 'quantile':
                from sklearn.preprocessing import QuantileTransformer
                scaler = QuantileTransformer(output_distribution='normal')
                data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
                self.scalers['quantile'] = scaler
        
        return data
    
    def dimensionality_reduction(self, data, method='pca', n_components=10):
        """Advanced dimensionality reduction techniques"""
        if isinstance(data, pd.DataFrame):
            numeric_data = data.select_dtypes(include=[np.number])
            
            if method == 'pca':
                reducer = PCA(n_components=n_components, random_state=42)
                reduced = reducer.fit_transform(numeric_data)
                self.dimensionality_reducers['pca'] = reducer
                
            elif method == 'ica':
                reducer = FastICA(n_components=n_components, random_state=42)
                reduced = reducer.fit_transform(numeric_data)
                self.dimensionality_reducers['ica'] = reducer
                
            elif method == 'tsne':
                reducer = TSNE(n_components=min(3, n_components), random_state=42)
                reduced = reducer.fit_transform(numeric_data)
                self.dimensionality_reducers['tsne'] = reducer
            
            # Create DataFrame with reduced components
            reduced_df = pd.DataFrame(
                reduced, 
                columns=[f'{method.upper()}_component_{i+1}' for i in range(reduced.shape[1])],
                index=data.index
            )
            
            return reduced_df
        
        return data
    
    def spatial_processing(self, gdf, target_crs='EPSG:4326'):
        """Advanced spatial data processing"""
        if isinstance(gdf, gpd.GeoDataFrame):
            # Reproject if needed
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)
            
            # Calculate spatial features
            gdf['area'] = gdf.geometry.area
            gdf['centroid'] = gdf.geometry.centroid
            gdf['centroid_lon'] = gdf.centroid.x
            gdf['centroid_lat'] = gdf.centroid.y
            
            # Spatial joins and proximity analysis
            if len(gdf) > 1:
                from scipy.spatial.distance import cdist
                coords = np.array([[geom.x, geom.y] for geom in gdf.centroid])
                distances = cdist(coords, coords)
                gdf['mean_distance_to_neighbors'] = np.mean(distances, axis=1)
                gdf['min_distance_to_neighbor'] = np.min(distances + np.eye(len(distances)) * 1e6, axis=1)
        
        return gdf
    
    def temporal_aggregation(self, data, time_col='date', freq='monthly'):
        """Advanced temporal aggregation and feature extraction"""
        if isinstance(data, pd.DataFrame) and time_col in data.columns:
            data[time_col] = pd.to_datetime(data[time_col])
            data = data.set_index(time_col)
            
            if freq == 'monthly':
                agg_data = data.resample('M').agg({
                    col: ['mean', 'std', 'min', 'max'] for col in data.select_dtypes(include=[np.number]).columns
                })
                
            elif freq == 'seasonal':
                agg_data = data.resample('Q').agg({
                    col: ['mean', 'std', 'min', 'max'] for col in data.select_dtypes(include=[np.number]).columns
                })
            
            # Flatten column names
            agg_data.columns = [f'{col[0]}_{col[1]}' for col in agg_data.columns]
            
            return agg_data.reset_index()
        
        return data
    
    def create_ml_features(self, data):
        """Create machine learning-ready features"""
        if isinstance(data, pd.DataFrame):
            ml_data = data.copy()
            
            # Polynomial features for key variables
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            key_cols = [col for col in numeric_cols if any(keyword in col.lower() 
                       for keyword in ['ph', 'carbon', 'temperature', 'precipitation'])]
            
            for col in key_cols[:3]:  # Limit to avoid too many features
                if col in ml_data.columns:
                    ml_data[f'{col}_squared'] = ml_data[col] ** 2
                    ml_data[f'{col}_log'] = np.log1p(np.abs(ml_data[col])) * np.sign(ml_data[col])
            
            # Interaction terms
            if len(key_cols) >= 2:
                for i in range(min(len(key_cols), 3)):
                    for j in range(i+1, min(len(key_cols), 3)):
                        col1, col2 = key_cols[i], key_cols[j]
                        ml_data[f'{col1}_x_{col2}'] = ml_data[col1] * ml_data[col2]
            
            return ml_data
        
        return data
