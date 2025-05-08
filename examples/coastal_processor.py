"""
Coastal Data Processor

This module provides functions to load and process coastal data,
properly inverting ocean/sea areas to identify landmass.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box

def load_coastal_data(parquet_path):
    """Load coastal data from parquet file"""
    print(f"Loading coastal data from: {parquet_path}")
    gdf = gpd.read_parquet(parquet_path)
    
    # Ensure CRS is set correctly
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    
    print(f"Loaded {len(gdf)} features with CRS: {gdf.crs}")
    return gdf

def extract_landmass_for_bbox(gdf, bbox):
    """
    Extract and invert coastal data for the specified bounding box.
    The parquet data represents ocean/sea areas, so we need to invert it to get landmass.
    
    Args:
        gdf: GeoDataFrame with ocean/sea polygons
        bbox: [min_lon, min_lat, max_lon, max_lat]
        
    Returns:
        GeoDataFrame with landmass polygons
    """
    # Create a shapely box for the bbox
    bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
    
    # Filter to features that intersect the bbox
    bbox_gdf = gdf[gdf.geometry.intersects(bbox_poly)].copy()
    
    # Clip features to the bbox
    bbox_gdf['geometry'] = bbox_gdf.geometry.intersection(bbox_poly)
    
    # Simplify geometries for better performance
    bbox_gdf['geometry'] = bbox_gdf.geometry.simplify(tolerance=0.001)
    
    print(f"Filtered to {len(bbox_gdf)} coastal features in the BBOX")
    
    # Create a unified ocean polygon (combining all features)
    ocean_poly = None
    if len(bbox_gdf) > 0:
        ocean_poly = bbox_gdf.geometry.unary_union
        
        # Create landmass by inverting (subtracting ocean from bbox)
        landmass_poly = bbox_poly.difference(ocean_poly)
        
        # Create landmass features
        if isinstance(landmass_poly, MultiPolygon):
            # Multiple landmass polygons
            landmass_parts = list(landmass_poly.geoms)
        else:
            # Single landmass polygon
            landmass_parts = [landmass_poly]
            
        # Create new GeoDataFrame with landmass features
        landmass_gdf = gpd.GeoDataFrame(
            {'feature_type': ['landmass'] * len(landmass_parts)},
            geometry=landmass_parts,
            crs=bbox_gdf.crs
        )
        
        # Filter out tiny polygons (noise)
        min_area = 0.00001  # Adjust as needed
        landmass_gdf = landmass_gdf[landmass_gdf.geometry.area > min_area]
        
        print(f"Created {len(landmass_gdf)} landmass features by inversion")
        return landmass_gdf
    else:
        # If no ocean features found, return empty GeoDataFrame
        print("No ocean features found in BBOX, returning empty landmass")
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

def visualize_landmass(landmass_gdf, bbox):
    """Create a visualization of landmass within bbox"""
    import folium
    
    # Calculate center of bbox
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles='CartoDB positron'
    )
    
    # Add landmass features
    folium.GeoJson(
        landmass_gdf,
        name='Landmass',
        style_function=lambda x: {
            'fillColor': '#8B4513',  # Brown for land
            'color': '#654321',
            'weight': 1,
            'fillOpacity': 0.7
        }
    ).add_to(m)
    
    # Add bbox outline
    folium.Rectangle(
        bounds=[(bbox[1], bbox[0]), (bbox[3], bbox[2])],
        color='red',
        weight=2,
        fill=False
    ).add_to(m)
    
    # Add title
    title_html = f'''
        <div style="position: fixed; top: 10px; left: 50px; width: 250px; 
                    background-color: white; padding: 10px; z-index: 9999; border-radius: 5px;">
            <h3 style="margin: 0;">Landmass Features</h3>
            <p style="margin: 5px 0 0 0;">{len(landmass_gdf)} features in selected area</p>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    folium.LayerControl().add_to(m)
    return m

# Example usage
if __name__ == "__main__":
    import os
    
    # Example path
    COASTAL_DATA_PATH = "../data/CoastalWatersAustraliaAndOceania/polygons_CoastalWatersAustraliaAndOceania.parquet"
    
    # Example bbox around Darwin
    DARWIN_BBOX = [130.5, -13.0, 131.2, -12.0]  # [min_lon, min_lat, max_lon, max_lat]
    
    if os.path.exists(COASTAL_DATA_PATH):
        # Load data
        coastal_gdf = load_coastal_data(COASTAL_DATA_PATH)
        
        # Extract landmass
        landmass_gdf = extract_landmass_for_bbox(coastal_gdf, DARWIN_BBOX)
        
        # Visualize
        landmass_map = visualize_landmass(landmass_gdf, DARWIN_BBOX)
        
        # Save map
        landmass_map.save("landmass_darwin.html")
        print("Map saved as landmass_darwin.html")
    else:
        print(f"Coastal data file not found: {COASTAL_DATA_PATH}")