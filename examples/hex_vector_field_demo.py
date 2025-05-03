"""
Geospatial Vector Field Demo using H3 Hexagonal Grid and Folium Maps

This example demonstrates how to visualize a maritime current vector field
using Uber's H3 hexagonal grid system and Folium for interactive maps.

Based on concepts from:
https://jens-wirelesscar.medium.com/geospatial-vector-fields-using-folium-maps-and-an-uber-h3-hexagonal-grid-a083b349b4cc
"""

import os
import sys
import numpy as np
import folium
from folium import plugins
import h3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import branca.colormap as cm

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def lat_lon_to_vector_components(lat, lon, center_lat=37.0, center_lon=-122.0):
    """
    Creates a realistic oceanic current pattern at the given lat/lon.
    This is a simplified model combining:
    1. A Gaussian vortex (cyclonic rotation)
    2. A general coastal flow
    
    Args:
        lat: Latitude
        lon: Longitude
        center_lat: Center of vortex latitude
        center_lon: Center of vortex longitude
        
    Returns:
        tuple: (u, v) vector components in m/s
    """
    # Convert to km for easier calculations
    km_per_degree_lat = 111.32  # km per degree of latitude
    # Longitude degrees to km depends on latitude
    km_per_degree_lon = 111.32 * np.cos(np.radians(lat))
    
    # Distance from vortex center in km
    lat_diff = (lat - center_lat) * km_per_degree_lat
    lon_diff = (lon - center_lon) * km_per_degree_lon
    dist = np.sqrt(lat_diff**2 + lon_diff**2)
    
    # Base coastal current (stronger near coast, weaker offshore)
    # Assuming coast is to the east, current flows southward
    coastal_strength = 0.5 * np.exp(-dist/50)  # Exponential decay with distance
    u_coastal = 0  # No east-west component
    v_coastal = -coastal_strength  # Southward flow
    
    # Vortex component (cyclonic rotation around center)
    vortex_strength = 1.0 * np.exp(-dist**2/(2*30**2))  # Gaussian profile
    if dist < 1e-6:
        u_vortex, v_vortex = 0, 0
    else:
        # Tangential velocity (positive = counterclockwise)
        u_vortex = -lat_diff/dist * vortex_strength  # Normalized and scaled
        v_vortex = lon_diff/dist * vortex_strength
    
    # Combined flow
    u = u_coastal + u_vortex
    v = v_coastal + v_vortex
    
    return u, v


def create_h3_hexagon_grid(center_lat, center_lon, radius_km=100, resolution=7):
    """
    Creates an H3 hexagonal grid around a center point.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_km: Radius in kilometers
        resolution: H3 resolution (0-15, higher is smaller hexagons)
        
    Returns:
        list: H3 indices for the hexagons in the grid
    """
    # Convert center point to H3 index
    center_hex = h3.latlng_to_cell(center_lat, center_lon, resolution)
    
    # Use a fixed number of rings based on resolution
    # Lower resolutions (larger hexagons) need fewer rings to cover the same area
    # This is a simplified approach - a better approach would compute this from 
    # average hexagon sizes at different resolutions
    ring_sizes = {
        5: 3,   # ~2.4km hexagons
        6: 7,   # ~1.1km hexagons
        7: 15,  # ~0.5km hexagons 
        8: 30,  # ~0.2km hexagons
        9: 60   # ~0.1km hexagons
    }
    num_rings = ring_sizes.get(resolution, int(radius_km / 0.5))  # Default: assume ~0.5km per ring
    
    # Get hexagons in the disk (k-ring)
    hexagons = h3.grid_disk(center_hex, num_rings)
    
    return list(hexagons)


def calculate_vector_field(hexagons, center_lat=37.0, center_lon=-122.0):
    """
    Calculate vector field values for all hexagons.
    
    Args:
        hexagons: List of H3 indices
        center_lat: Center of vector field pattern
        center_lon: Center of vector field pattern
        
    Returns:
        DataFrame with hexagon indices, centers, and vector components
    """
    hex_data = []
    
    for h3_idx in hexagons:
        # Get the center point of the hexagon
        lat, lon = h3.cell_to_latlng(h3_idx)
        
        # Calculate vector components
        u, v = lat_lon_to_vector_components(lat, lon, center_lat, center_lon)
        
        # Calculate magnitude for color mapping
        magnitude = np.sqrt(u**2 + v**2)
        
        # Calculate direction for the arrow
        direction = np.degrees(np.arctan2(v, u))
        
        # Store the data
        hex_data.append({
            'h3_index': h3_idx,
            'lat': lat,
            'lon': lon,
            'u': u,
            'v': v,
            'magnitude': magnitude,
            'direction': direction
        })
    
    return pd.DataFrame(hex_data)


def add_hex_grid_vectors_to_map(m, hex_data, scale=0.25, arrow_color='speed'):
    """
    Add vector arrows to a folium map for the hexagonal grid using SVG arrowheads.
    
    Args:
        m: Folium map object
        hex_data: DataFrame with hexagon data and vector components
        scale: Scaling factor for arrows
        arrow_color: 'speed' for speed-colored arrows, or a specific color
    """
    # Create a colormap for the magnitudes
    if arrow_color == 'speed':
        colormap = cm.linear.YlOrRd_09.scale(
            hex_data['magnitude'].min(),
            hex_data['magnitude'].max()
        )
    
    # Add a custom CSS to define the arrowhead styles
    arrowhead_css = """
    <style>
        .hex-arrow {
            pointer-events: none;
            transform-origin: center;
        }
        .hex-arrowhead {
            fill-opacity: 0.9;
            stroke-width: 1;
        }
    </style>
    """
    m.get_root().header.add_child(folium.Element(arrowhead_css))
    
    # Function to create an SVG arrow with an arrowhead
    def create_arrow_svg(angle, magnitude, color, max_magnitude):
        # Scale arrow length based on magnitude and relative to max value
        # Cap the length to ensure it stays within the hex
        relative_magnitude = magnitude / max_magnitude if max_magnitude > 0 else 0.5
        length = min(20 * relative_magnitude, 10)  # Limit size to 10px
        
        # Minimum size so very small currents are still visible
        if length < 3:
            length = 3
        
        # Calculate arrowhead size proportional to length
        arrowhead_width = min(6, length * 0.6)
        arrowhead_length = min(6, length * 0.8)
        
        # Calculate arrow end points
        end_x = length
        end_y = 0
        
        # Define the SVG path for the arrow
        svg = f"""
        <svg width="24" height="24" viewBox="-12 -12 24 24" class="hex-arrow" 
             style="transform: rotate({angle}deg);">
            <defs>
                <marker id="arrowhead_{int(magnitude*100)}" 
                        viewBox="0 0 10 10" refX="1" refY="5"
                        markerWidth="{arrowhead_width}" markerHeight="{arrowhead_width}" 
                        orient="auto-start-reverse">
                    <path d="M 0 0 L 10 5 L 0 10 z" class="hex-arrowhead" 
                          fill="{color}" stroke="{color}" />
                </marker>
            </defs>
            <line x1="0" y1="0" x2="{end_x}" y2="{end_y}" 
                  stroke="{color}" stroke-width="2" 
                  marker-end="url(#arrowhead_{int(magnitude*100)})" />
        </svg>
        """
        return svg
    
    # Find the maximum magnitude for scaling
    max_magnitude = hex_data['magnitude'].max()
    
    # Add one arrow for each hex center
    for idx, row in hex_data.iterrows():
        # Determine color based on magnitude
        if arrow_color == 'speed':
            # Normalize magnitude to 0-1 range
            mag_norm = (row['magnitude'] - hex_data['magnitude'].min()) / \
                      (hex_data['magnitude'].max() - hex_data['magnitude'].min() + 1e-10)
            
            # Create a color scale from yellow to red
            r = 255
            g = int(255 * (1 - mag_norm))
            b = 0
            color = f"#{r:02x}{g:02x}{b:02x}"
        else:
            color = arrow_color
        
        # Create the SVG arrow
        arrow_svg = create_arrow_svg(row['direction'], row['magnitude'], color, max_magnitude)
        
        # Create a custom icon with the SVG arrow
        icon = folium.DivIcon(
            icon_size=(24, 24),  # Fixed size container
            icon_anchor=(12, 12),  # Center point
            html=arrow_svg,
            class_name="hex-vector-arrow"
        )
        
        # Add a marker
        folium.Marker(
            location=[row['lat'], row['lon']],
            icon=icon,
            tooltip=f"Speed: {row['magnitude']:.2f} m/s<br>Direction: {row['direction']:.0f}°"
        ).add_to(m)
    
    # Add colorbar legend if using speed colors
    if arrow_color == 'speed':
        colormap.caption = 'Current Speed (m/s)'
        colormap.add_to(m)
        
    # Add a legend for the arrows
    legend_html = '''
        <div style="position: fixed; 
            bottom: 50px; right: 50px; width: 200px;
            border: 2px solid grey; z-index: 9999; background-color: white;
            padding: 10px;
            font-size: 14px; line-height: 1.5;">
            <div style="font-weight: bold; margin-bottom: 5px;">Maritime Currents</div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="width: 24px; height: 24px; position: relative;">
                    <svg width="24" height="24" viewBox="-12 -12 24 24">
                        <line x1="0" y1="0" x2="10" y2="0" stroke="#ff0000" stroke-width="2" />
                        <polygon points="10,0 6,3 6,-3" fill="#ff0000" stroke="#ff0000" />
                    </svg>
                </div>
                <span style="margin-left: 5px;">Stronger current</span>
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="width: 24px; height: 24px; position: relative;">
                    <svg width="24" height="24" viewBox="-12 -12 24 24">
                        <line x1="0" y1="0" x2="10" y2="0" stroke="#ffcc00" stroke-width="2" />
                        <polygon points="10,0 6,3 6,-3" fill="#ffcc00" stroke="#ffcc00" />
                    </svg>
                </div>
                <span style="margin-left: 5px;">Weaker current</span>
            </div>
            <div style="margin-top: 5px; font-style: italic; font-size: 12px;">
                Arrow direction indicates flow direction
            </div>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))


def add_hex_grid_to_map(m, hexagons, color='blue', fill_color='blue', fill_opacity=0.1):
    """
    Add hexagon grid boundaries to a folium map.
    
    Args:
        m: Folium map object
        hexagons: List of H3 indices
        color: Border color
        fill_color: Fill color
        fill_opacity: Fill opacity
    """
    for h3_idx in hexagons:
        # Get the boundary vertices
        boundary = h3.cell_to_boundary(h3_idx)
        # Convert to (lat, lon) format for folium
        boundary = [[lat, lon] for lat, lon in boundary]
        
        # Add the polygon to the map
        folium.Polygon(
            locations=boundary,
            color=color,
            weight=1,
            fill=True,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            opacity=0.5
        ).add_to(m)


def main():
    # Define the center of our area of interest (Monterey Bay, CA)
    center_lat = 36.8
    center_lon = -122.0
    
    # Create an H3 hexagonal grid
    print("Creating H3 hexagonal grid...")
    hexagons = create_h3_hexagon_grid(center_lat, center_lon, 
                                      radius_km=50, resolution=7)
    print(f"Created {len(hexagons)} hexagons")
    
    # Calculate vector field for each hexagon
    print("Calculating vector field...")
    hex_data = calculate_vector_field(hexagons, center_lat, center_lon)
    
    # Create a folium map centered on our area
    print("Creating map...")
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles='CartoDB positron'
    )
    
    # Add the hexagon grid boundaries
    print("Adding hexagon grid...")
    add_hex_grid_to_map(m, hexagons)
    
    # Add the vector field arrows
    print("Adding vector field arrows...")
    add_hex_grid_vectors_to_map(m, hex_data, scale=50, arrow_color='speed')
    
    # Add a title
    title_html = '''
         <h3 align="center" style="font-size:16px"><b>Maritime Current Vector Field</b></h3>
         <h4 align="center" style="font-size:12px"><i>Using H3 Hexagonal Grid</i></h4>
         '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map to an HTML file
    output_file = "hex_vector_field.html"
    m.save(output_file)
    print(f"Map saved to {output_file}")
    
    # Optional: Create a static version using matplotlib for the report
    print("Creating static visualization...")
    plt.figure(figsize=(12, 10))
    
    # Extract hex centers
    lats = hex_data['lat'].values
    lons = hex_data['lon'].values
    u = hex_data['u'].values
    v = hex_data['v'].values
    magnitude = hex_data['magnitude'].values
    
    # Create colormap
    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(magnitude.min(), magnitude.max())
    
    # Plot arrows
    plt.quiver(lons, lats, u, v, magnitude, 
               cmap=cmap, norm=norm, 
               scale=50, width=0.002, 
               pivot='tail', zorder=10)
    
    # Plot hexagon boundaries for a few hexagons (not all to avoid clutter)
    sample_hexes = np.random.choice(hexagons, min(100, len(hexagons)//5))
    for h3_idx in sample_hexes:
        boundary = h3.cell_to_boundary(h3_idx)
        boundary = np.array(boundary)
        plt.plot(boundary[:, 1], boundary[:, 0], 'b-', alpha=0.2, linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Current Speed (m/s)')
    
    # Configure plot
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Maritime Current Vector Field on H3 Hexagonal Grid')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Save the static version
    plt.savefig("hex_vector_field_static.png", dpi=300, bbox_inches='tight')
    print("Static visualization saved to hex_vector_field_static.png")
    
    print("\nNext steps would include implementing hierarchical path planning using this hexagonal grid.")


if __name__ == "__main__":
    main()