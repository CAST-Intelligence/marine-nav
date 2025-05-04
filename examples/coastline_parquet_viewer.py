"""
Interactive Coastline Visualization from GeoParquet File

This script creates an interactive web visualization of coastline data
from a GeoParquet file using Folium.
"""

import os
import sys
import folium
import geopandas as gpd
import pandas as pd
import numpy as np
from folium.plugins import Draw, MeasureControl

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    # Path to the GeoParquet file
    parquet_file = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../data/CoastalWatersAustraliaAndOceania/polygons_CoastalWatersAustraliaAndOceania.parquet'
    ))
    
    # Load the GeoParquet file
    print(f"Loading GeoParquet file: {parquet_file}")
    gdf = gpd.read_parquet(parquet_file)
    
    # Set CRS if it's missing - assuming WGS 84 (EPSG:4326)
    if gdf.crs is None:
        print("CRS is missing, setting to EPSG:4326 (WGS 84)")
        gdf.set_crs(epsg=4326, inplace=True)
    
    print(f"CRS: {gdf.crs}")
    print(f"Number of features: {len(gdf)}")
    print(f"Columns: {gdf.columns.tolist()}")
    
    # Calculate the centroid of all geometries to use as map center
    centroid = gdf.geometry.union_all().centroid
    center_lat, center_lon = centroid.y, centroid.x
    
    # Create a folium map
    print("Creating interactive map...")
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='CartoDB positron'
    )
    
    # Identify coastal areas by geometric complexity
    print("Identifying coastal areas by geometric shape complexity...")
    
    # Calculate complexity metrics to identify coastal tiles
    # Complexity = perimeter / (2 * sqrt(π * area))
    # Higher values = more complex/irregular shapes = likely coastal
    gdf['area'] = gdf.geometry.area
    gdf['perimeter'] = gdf.geometry.boundary.length
    gdf['complexity'] = gdf['perimeter'] / (2 * np.sqrt(np.pi * gdf['area']))
    
    # Get complexity statistics
    complexity_stats = gdf['complexity'].describe()
    print(f"Complexity stats: min={complexity_stats['min']:.2f}, mean={complexity_stats['mean']:.2f}, max={complexity_stats['max']:.2f}")
    
    # Use a much lower threshold to capture more potential coastal areas
    # This ensures better coverage of the entire coastline
    threshold = complexity_stats['mean'] + 0.1 * complexity_stats['std']
    
    # Filter for somewhat complex shapes
    coastal_gdf = gdf[gdf['complexity'] > threshold].copy()
    print(f"Identified {len(coastal_gdf)} likely coastal areas out of {len(gdf)} total features")
    
    # If we have too few, also include some smaller areas that might be coastal inlets
    if len(coastal_gdf) < 2000:
        # Also include small area features which are likely to be coastal inlets
        small_areas = gdf[gdf['area'] < gdf['area'].quantile(0.25)].copy()
        print(f"Adding {len(small_areas)} small area features that may be coastal inlets")
        coastal_gdf = pd.concat([coastal_gdf, small_areas]).drop_duplicates()
    
    # If we still have too many, prioritize by a combination of complexity and location
    max_features = 4000  # Increased limit for better coverage
    if len(coastal_gdf) > max_features:
        print(f"Sampling {max_features} features for better performance...")
        
        # Sort by complexity but also ensure geographic distribution
        # We'll divide into regions and take top features from each region
        coastal_gdf['lat_bin'] = pd.cut(coastal_gdf.geometry.centroid.y, 10)
        coastal_gdf['lon_bin'] = pd.cut(coastal_gdf.geometry.centroid.x, 10)
        
        # Group by region and select top features from each
        result = []
        for _, group in coastal_gdf.groupby(['lat_bin', 'lon_bin']):
            # Take more features from groups with more complexity
            n_features = max(5, min(300, len(group)))
            result.append(group.sort_values('complexity', ascending=False).head(n_features))
        
        coastal_gdf = pd.concat(result)
        print(f"Selected {len(coastal_gdf)} features distributed across regions")
    
    # Simplify geometries to reduce file size and improve performance
    print("Simplifying geometries...")
    gdf_simplified = coastal_gdf.copy()
    tolerance = 0.005  # Balance between detail and performance
    gdf_simplified['geometry'] = gdf_simplified.geometry.simplify(tolerance)
    
    # Use a subset of columns if there are many
    columns_to_show = gdf_simplified.columns.tolist()[:5]  # First 5 columns
    
    # Add the geometries to the map
    print("Adding geometries to map...")
    folium.GeoJson(
        gdf_simplified,
        name='Coastal Waters',
        style_function=lambda x: {
            # Color based on complexity (more complex = more coastal = darker blue)
            'fillColor': '#0096c7',  # Light blue for coastal areas
            'color': '#023e8a',      # Dark blue outline
            'weight': 1,
            'fillOpacity': 0.4
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['complexity', 'area', 'perimeter'], 
            aliases=['Coastal Complexity', 'Area', 'Perimeter'],
            localize=True
        )
    ).add_to(m)
    
    # Add drawing tools
    Draw(
        export=True,
        position='topleft',
        draw_options={
            'polyline': True,
            'polygon': True,
            'rectangle': True,
            'circle': True,
            'marker': True,
            'circlemarker': False
        }
    ).add_to(m)
    
    # Add measurement tools
    MeasureControl(
        position='bottomleft',
        primary_length_unit='kilometers',
        secondary_length_unit='miles',
        primary_area_unit='square kilometers',
        secondary_area_unit='acres'
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add a title
    title_html = f'''
         <h3 align="center" style="font-size:16px"><b>Coastal Waters: Australia and Oceania</b></h3>
         <h4 align="center" style="font-size:12px"><i>Comprehensive coastal view with {len(gdf_simplified)} features selected by shape analysis</i></h4>
         '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map to an HTML file
    output_file = "coastline_visualization.html"
    m.save(output_file)
    print(f"Map saved to {output_file}")
    
    # Also return the map object so it can be displayed in Jupyter notebook if needed
    return m

if __name__ == "__main__":
    main()