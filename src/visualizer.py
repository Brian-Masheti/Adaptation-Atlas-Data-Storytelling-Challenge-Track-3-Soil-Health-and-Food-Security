"""
Advanced Interactive Visualization Suite for Soil Health and Climate Analysis
State-of-the-art visualizations using Plotly, D3.js, and modern web technologies
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import geopandas as gpd
import folium
from folium import plugins
import warnings
warnings.filterwarnings('ignore')

class AdvancedVisualizer:
    """
    Advanced visualization suite for creating interactive, publication-quality
    visualizations for soil health and climate data storytelling
    """
    
    def __init__(self):
        self.color_schemes = {
            'soil_health': ['#2E7D32', '#43A047', '#66BB6A', '#81C784', '#A5D6A7'],
            'climate_stress': ['#B71C1C', '#C62828', '#D32F2F', '#E53935', '#F44336'],
            'risk_levels': ['#4CAF50', '#FFC107', '#FF5722', '#D32F2F'],
            'adaptation': ['#1976D2', '#2196F3', '#03A9F4', '#00BCD4', '#009688']
        }
        
    def create_interactive_soil_health_map(self, gdf, health_score_col='health_index'):
        """Create interactive soil health map with Plotly"""
        if isinstance(gdf, gpd.GeoDataFrame) and health_score_col in gdf.columns:
            # Convert to GeoJSON for Plotly
            gdf['centroid_lon'] = gdf.geometry.centroid.x
            gdf['centroid_lat'] = gdf.geometry.centroid.y
            
            # Create base map
            fig = go.Figure()
            
            # Add soil health choropleth
            fig.add_trace(go.Choroplethmapbox(
                geojson=gdf.geometry.__geo_interface__,
                locations=gdf.index,
                z=gdf[health_score_col],
                colorscale='RdYlGn',
                reversescale=True,
                marker_opacity=0.7,
                marker_line_width=0.5,
                colorbar=dict(
                    title="Soil Health Index",
                    titleside="right",
                    tickmode="linear",
                    tick0=0,
                    dtick=0.2
                ),
                hovertemplate='<b>%{text}</b><br>' +
                             'Soil Health: %{z:.3f}<br>' +
                             '<extra></extra>',
                text=gdf.index
            ))
            
            # Add centroids for detailed information
            fig.add_trace(go.Scattermapbox(
                lat=gdf['centroid_lat'],
                lon=gdf['centroid_lon'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=gdf[health_score_col],
                    colorscale='RdYlGn',
                    reversescale=True,
                    showscale=False,
                    sizemode='diameter'
                ),
                text=gdf.index,
                hovertemplate='<b>%{text}</b><br>' +
                             'Health Score: %{marker.color:.3f}<br>' +
                             '<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(
                    center=dict(lat=gdf['centroid_lat'].mean(), lon=gdf['centroid_lon'].mean()),
                    zoom=6
                ),
                title={
                    'text': 'Interactive Soil Health Map - Sub-Saharan Africa',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#2E7D32'}
                },
                height=600,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            return fig
        
        return None
    
    def create_climate_stress_animation(self, climate_data, temporal_col='date'):
        """Create animated climate stress visualization"""
        if isinstance(climate_data, pd.DataFrame):
            # Prepare data for animation
            if temporal_col not in climate_data.columns:
                climate_data = climate_data.reset_index()
                temporal_col = climate_data.columns[0]
            
            # Create animated map
            fig = go.Figure()
            
            # Add animation frames for each time period
            frames = []
            unique_times = climate_data[temporal_col].unique()[:50]  # Limit to 50 frames for performance
            
            for time_val in unique_times:
                time_data = climate_data[climate_data[temporal_col] == time_val]
                
                frame = go.Frame(
                    data=[
                        go.Scattermapbox(
                            lat=time_data['lat'],
                            lon=time_data['lon'],
                            mode='markers',
                            marker=dict(
                                size=time_data['combined_stress'] * 20,
                                color=time_data['combined_stress'],
                                colorscale='Reds',
                                cmin=0,
                                cmax=1,
                                colorbar=dict(title="Climate Stress Index"),
                                opacity=0.8
                            ),
                            text=time_data.index,
                            hovertemplate='Location: %{text}<br>' +
                                         'Stress Index: %{marker.color:.3f}<br>' +
                                         'Date: ' + str(time_val) + '<br>' +
                                         '<extra></extra>'
                        )
                    ],
                    name=str(time_val)
                )
                frames.append(frame)
            
            # Add initial frame
            initial_data = climate_data[climate_data[temporal_col] == unique_times[0]]
            fig.add_trace(go.Scattermapbox(
                lat=initial_data['lat'],
                lon=initial_data['lon'],
                mode='markers',
                marker=dict(
                    size=initial_data['combined_stress'] * 20,
                    color=initial_data['combined_stress'],
                    colorscale='Reds',
                    cmin=0,
                    cmax=1,
                    colorbar=dict(title="Climate Stress Index"),
                    opacity=0.8
                ),
                text=initial_data.index,
                hovertemplate='Location: %{text}<br>' +
                             'Stress Index: %{marker.color:.3f}<br>' +
                             '<extra></extra>'
            ))
            
            # Add animation controls
            fig.frames = frames
            fig.update_layout(
                updatemenus=[dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(args=[None, {"frame": {"duration": 500, "redraw": True},
                                         "fromcurrent": True}]),
                             label="Play",
                             method="animate"
                        ),
                        dict(args=[None, {"frame": {"duration": 0, "redraw": True},
                                         "fromcurrent": True}]),
                             label="Pause",
                             method="animate"
                        )
                    ]),
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.011,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                )],
                
                sliders=[dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue={"prefix": "Time: "},
                    pad={"b": 10, "t": 50},
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[dict(args=[
                        [str(time_val)],
                        {"frame": {"duration": 300, "redraw": True},
                         "mode": "immediate",
                         "transition": {"duration": 300}}
                    ], label=str(time_val), method="animate") for time_val in unique_times]
                )],
                
                mapbox_style="open-street-map",
                mapbox=dict(
                    center=dict(lat=climate_data['lat'].mean(), lon=climate_data['lon'].mean()),
                    zoom=5
                ),
                title={
                    'text': 'Climate Stress Evolution Animation',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#D32F2F'}
                },
                height=600
            )
            
            return fig
        
        return None
    
    def create_compound_risk_dashboard(self, risk_data, agricultural_data):
        """Create comprehensive compound risk dashboard"""
        if isinstance(risk_data, pd.DataFrame) and isinstance(agricultural_data, pd.DataFrame):
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Compound Risk Distribution', 'Risk vs Agricultural Value',
                              'Soil vs Climate Risk', 'Risk Category Breakdown',
                              'Economic Exposure by Risk Level', 'Temporal Risk Trend'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "domain"}],
                       [{"secondary_y": False}, {"secondary_y": False}]],
                vertical_spacing=0.08,
                horizontal_spacing=0.1
            )
            
            # 1. Compound Risk Distribution
            fig.add_trace(
                go.Histogram(x=risk_data['compound_risk_multiplicative'],
                           nbinsx=30,
                           name='Risk Distribution',
                           marker_color='#FF5722',
                           opacity=0.7),
                row=1, col=1
            )
            
            # 2. Risk vs Agricultural Value
            if 'economic_value' in agricultural_data.columns:
                fig.add_trace(
                    go.Scatter(x=risk_data['compound_risk_multiplicative'],
                             y=agricultural_data['economic_value'],
                             mode='markers',
                             name='Risk vs Value',
                             marker=dict(
                                 color=risk_data['compound_risk_multiplicative'],
                                 colorscale='Reds',
                                 size=8,
                                 opacity=0.7
                             ),
                             hovertemplate='Risk: %{x:.3f}<br>Value: %{y:,.0f}<br><extra></extra>'),
                    row=1, col=2
                )
            
            # 3. Soil vs Climate Risk Scatter
            fig.add_trace(
                go.Scatter(x=risk_data['soil_risk_index'],
                         y=risk_data['climate_risk_index'],
                         mode='markers',
                         name='Soil vs Climate Risk',
                         marker=dict(
                             color=risk_data['compound_risk_multiplicative'],
                             colorscale='Viridis',
                             size=10,
                             opacity=0.7,
                             colorbar=dict(title="Compound Risk")
                         ),
                         hovertemplate='Soil Risk: %{x:.3f}<br>Climate Risk: %{y:.3f}<br>Compound: %{marker.color:.3f}<br><extra></extra>'),
                row=2, col=1
            )
            
            # 4. Risk Category Pie Chart
            if 'risk_category' in risk_data.columns:
                risk_counts = risk_data['risk_category'].value_counts()
                fig.add_trace(
                    go.Pie(labels=risk_counts.index,
                          values=risk_counts.values,
                          name='Risk Categories',
                          marker_colors=['#4CAF50', '#FFC107', '#FF5722']),
                    row=2, col=2
                )
            
            # 5. Economic Exposure by Risk Level
            if 'economic_value' in agricultural_data.columns:
                high_risk_value = agricultural_data.loc[risk_data['compound_risk_multiplicative'] > 0.7, 'economic_value'].sum()
                medium_risk_value = agricultural_data.loc[(risk_data['compound_risk_multiplicative'] > 0.4) & 
                                                        (risk_data['compound_risk_multiplicative'] <= 0.7), 'economic_value'].sum()
                low_risk_value = agricultural_data.loc[risk_data['compound_risk_multiplicative'] <= 0.4, 'economic_value'].sum()
                
                fig.add_trace(
                    go.Bar(x=['High Risk', 'Medium Risk', 'Low Risk'],
                          y=[high_risk_value, medium_risk_value, low_risk_value],
                          name='Economic Exposure',
                          marker_color=['#D32F2F', '#FF5722', '#FFC107']),
                    row=3, col=1
                )
            
            # 6. Temporal Risk Trend (if temporal data available)
            if len(risk_data) > 10:
                fig.add_trace(
                    go.Scatter(x=list(range(len(risk_data))),
                             y=risk_data['compound_risk_multiplicative'].rolling(window=5).mean(),
                             mode='lines',
                             name='Risk Trend',
                             line=dict(color='#2196F3', width=3)),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text='Comprehensive Compound Risk Analysis Dashboard',
                    x=0.5,
                    xanchor='center',
                    font=dict(size=24, color='#1976D2')
                ),
                height=1200,
                showlegend=True,
                template='plotly_white'
            )
            
            return fig
        
        return None
    
    def create_adaptation_priority_visualization(self, priorities_data):
        """Create interactive adaptation priority visualization"""
        if isinstance(priorities_data, dict) and 'recommendations' in priorities_data:
            recommendations = priorities_data['recommendations']
            
            if recommendations:
                # Create DataFrame for easier manipulation
                df = pd.DataFrame(recommendations)
                
                # Create priority map
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Priority Score Distribution', 'Cost-Benefit Analysis',
                                  'Recommended Actions Frequency', 'Risk Type Distribution'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"type": "domain"}, {"type": "domain"}]],
                    vertical_spacing=0.15,
                    horizontal_spacing=0.1
                )
                
                # 1. Priority Score Distribution
                fig.add_trace(
                    go.Histogram(x=df['priority_score'],
                               nbinsx=20,
                               name='Priority Scores',
                               marker_color='#1976D2',
                               opacity=0.7),
                    row=1, col=1
                )
                
                # 2. Cost-Benefit Scatter
                fig.add_trace(
                    go.Scatter(x=df['estimated_cost'],
                             y=df['expected_benefit'],
                             mode='markers',
                             name='Cost-Benefit Analysis',
                             marker=dict(
                                 size=df['priority_score'] * 100,
                                 color=df['priority_score'],
                                 colorscale='Viridis',
                                 opacity=0.7,
                                 colorbar=dict(title="Priority Score")
                             ),
                             text=df['location_id'],
                             hovertemplate='Location: %{text}<br>Cost: %{x:,.0f}<br>Benefit: %{y:,.0f}<br>Priority: %{marker.color:.3f}<br><extra></extra>'),
                    row=1, col=2
                )
                
                # Add 45-degree line for reference
                max_val = max(df['estimated_cost'].max(), df['expected_benefit'].max())
                fig.add_trace(
                    go.Scatter(x=[0, max_val], y=[0, max_val],
                             mode='lines',
                             name='Break-even Line',
                             line=dict(dash='dash', color='red')),
                    row=1, col=2
                )
                
                # 3. Recommended Actions Frequency
                all_actions = []
                for actions in df['recommended_actions']:
                    all_actions.extend(actions)
                
                action_counts = pd.Series(all_actions).value_counts().head(10)
                fig.add_trace(
                    go.Pie(labels=action_counts.index,
                          values=action_counts.values,
                          name='Recommended Actions',
                          marker_colors=px.colors.qualitative.Set3),
                    row=2, col=1
                )
                
                # 4. Risk Type Distribution
                all_risks = []
                for risks in df['dominant_risks']:
                    all_risks.extend(risks)
                
                risk_counts = pd.Series(all_risks).value_counts()
                fig.add_trace(
                    go.Pie(labels=risk_counts.index,
                          values=risk_counts.values,
                          name='Risk Types',
                          marker_colors=['#FF5722', '#2196F3']),
                    row=2, col=2
                )
                
                # Update layout
                fig.update_layout(
                    title=dict(
                        text='Adaptation Priorities Analysis Dashboard',
                        x=0.5,
                        xanchor='center',
                        font=dict(size=24, 'color': '#009688')
                    ),
                    height=800,
                    showlegend=True,
                    template='plotly_white'
                )
                
                return fig
        
        return None
    
    def create_soil_health_trend_analysis(self, temporal_data):
        """Create comprehensive soil health trend analysis"""
        if isinstance(temporal_data, pd.DataFrame):
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Soil Health Index Trend', 'Component Trends',
                              'Seasonal Decomposition', 'Year-over-Year Change'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # 1. Overall Soil Health Trend
            if 'health_index' in temporal_data.columns:
                fig.add_trace(
                    go.Scatter(x=temporal_data.index,
                             y=temporal_data['health_index'],
                             mode='lines',
                             name='Health Index',
                             line=dict(color='#4CAF50', width=3)),
                    row=1, col=1
                )
                
                # Add trend line
                x_numeric = np.arange(len(temporal_data))
                trend_coeffs = np.polyfit(x_numeric, temporal_data['health_index'].dropna(), 1)
                trend_line = np.poly1d(trend_coeffs)(x_numeric)
                
                fig.add_trace(
                    go.Scatter(x=temporal_data.index,
                             y=trend_line,
                             mode='lines',
                             name='Trend Line',
                             line=dict(dash='dash', color='red')),
                    row=1, col=1
                )
            
            # 2. Component Trends
            soil_components = ['ph_score', 'organic_carbon_score', 'texture_score', 'erosion_score']
            colors = ['#2196F3', '#FF9800', '#9C27B0', '#F44336']
            
            for i, component in enumerate(soil_components):
                if component in temporal_data.columns:
                    fig.add_trace(
                        go.Scatter(x=temporal_data.index,
                                 y=temporal_data[component],
                                 mode='lines',
                                 name=component.replace('_', ' ').title(),
                                 line=dict(color=colors[i], width=2)),
                        row=1, col=2
                    )
            
            # 3. Seasonal Decomposition (simplified)
            if 'health_index' in temporal_data.columns:
                # Calculate moving averages for trend and seasonal components
                trend = temporal_data['health_index'].rolling(window=12, center=True).mean()
                seasonal = temporal_data['health_index'] - trend
                
                fig.add_trace(
                    go.Scatter(x=temporal_data.index,
                             y=seasonal,
                             mode='lines',
                             name='Seasonal Component',
                             line=dict(color='#FF5722', width=2)),
                    row=2, col=1
                )
            
            # 4. Year-over-Year Change
            if 'health_index' in temporal_data.columns:
                yoy_change = temporal_data['health_index'].pct_change(periods=12) * 100
                
                fig.add_trace(
                    go.Scatter(x=temporal_data.index,
                             y=yoy_change,
                             mode='lines',
                             name='YoY Change (%)',
                             line=dict(color='#607D8B', width=2)),
                    row=2, col=2
                )
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text='Soil Health Trend Analysis Dashboard',
                    x=0.5,
                    xanchor='center',
                    font=dict(size=24, color='#2E7D32')
                ),
                height=800,
                showlegend=True,
                template='plotly_white'
            )
            
            return fig
        
        return None
    
    def create_climate_projection_visualization(self, projections_data):
        """Create climate projection scenario visualization"""
        if isinstance(projections_data, dict) and 'future_projections' in projections_data:
            future_data = projections_data['future_projections']
            
            if future_data:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Temperature Projections', 'Precipitation Projections',
                                  'Stress Index Evolution', 'Scenario Comparison'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]],
                    vertical_spacing=0.12,
                    horizontal_spacing=0.1
                )
                
                # Create time index for future projections
                time_index = pd.date_range(start='2024', periods=len(list(future_data.values())[0]), freq='M')
                
                # 1. Temperature Projections
                if 'temperature' in future_data:
                    fig.add_trace(
                        go.Scatter(x=time_index,
                                 y=future_data['temperature'],
                                 mode='lines',
                                 name='Temperature',
                                 line=dict(color='#D32F2F', width=3)),
                        row=1, col=1
                    )
                
                # 2. Precipitation Projections
                if 'precipitation' in future_data:
                    fig.add_trace(
                        go.Scatter(x=time_index,
                                 y=future_data['precipitation'],
                                 mode='lines',
                                 name='Precipitation',
                                 line=dict(color='#1976D2', width=3)),
                        row=1, col=2
                    )
                
                # 3. Stress Index Evolution
                if 'future_stress_indices' in projections_data:
                    stress_data = projections_data['future_stress_indices']
                    if 'combined_stress' in stress_data:
                        fig.add_trace(
                            go.Scatter(x=time_index,
                                     y=stress_data['combined_stress'],
                                     mode='lines',
                                     name='Combined Stress',
                                     line=dict(color='#FF5722', width=3)),
                            row=2, col=1
                        )
                
                # 4. Scenario Comparison (placeholder for multiple scenarios)
                scenarios = ['ssp1', 'ssp3', 'ssp5']
                colors = ['#4CAF50', '#FFC107', '#D32F2F']
                
                for i, scenario in enumerate(scenarios):
                    # Generate placeholder scenario data
                    scenario_data = np.random.normal(0.5, 0.1, len(time_index))
                    fig.add_trace(
                        go.Scatter(x=time_index,
                                 y=scenario_data,
                                 mode='lines',
                                 name=f'Scenario {scenario.upper()}',
                                 line=dict(color=colors[i], width=2, dash='dash')),
                        row=2, col=2
                    )
                
                # Update layout
                fig.update_layout(
                    title=dict(
                        text='Climate Stress Projections Dashboard',
                        x=0.5,
                        xanchor='center',
                        font=dict(size=24, color='#D32F2F')
                    ),
                    height=800,
                    showlegend=True,
                    template='plotly_white'
                )
                
                return fig
        
        return None
    
    def create_folium_interactive_map(self, gdf, risk_col='compound_risk_multiplicative'):
        """Create interactive Folium map for advanced spatial analysis"""
        if isinstance(gdf, gpd.GeoDataFrame) and risk_col in gdf.columns:
            # Calculate center point
            center_lat = gdf.geometry.centroid.y.mean()
            center_lon = gdf.geometry.centroid.x.mean()
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=6,
                tiles='OpenStreetMap'
            )
            
            # Add choropleth layer
            folium.Choropleth(
                geo_data=gdf,
                data=gdf,
                columns=[gdf.index.name or 'index', risk_col],
                key_on='feature.id',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name='Compound Risk Index'
            ).add_to(m)
            
            # Add markers for high-risk areas
            high_risk = gdf[gdf[risk_col] > 0.7]
            for idx, row in high_risk.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.centroid.y, row.geometry.centroid.x],
                    radius=10,
                    popup=f'Location: {idx}<br>Risk: {row[risk_col]:.3f}',
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.7
                ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Add fullscreen button
            plugins.Fullscreen().add_to(m)
            
            return m
        
        return None
    
    def export_visualizations(self, figs, output_dir='visualizations'):
        """Export visualizations to multiple formats"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = []
        
        for name, fig in figs.items():
            if fig is not None:
                # Export as HTML
                html_file = os.path.join(output_dir, f'{name}.html')
                fig.write_html(html_file)
                exported_files.append(html_file)
                
                # Export as PNG (if possible)
                try:
                    png_file = os.path.join(output_dir, f'{name}.png')
                    fig.write_image(png_file, width=1200, height=800)
                    exported_files.append(png_file)
                except Exception as e:
                    print(f"Could not export {name} as PNG: {e}")
        
        return exported_files
