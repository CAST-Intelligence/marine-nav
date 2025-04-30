#!/usr/bin/env python3
"""
Tidal channel demonstration showing water flow changes over time.
This version generates static images rather than interactive animations.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.animation as animation

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.spatio_temporal_field import SpatioTemporalField
from src.core.temporal_grid import TemporalNavigationGrid


def create_tidal_channel(
    grid_size=(60, 30), 
    time_steps=12,
    x_range=(0, 100),
    y_range=(0, 50),
    tidal_period=12.42  # Tidal period in hours (M2 tide)
):
    """Create a time-varying tidal channel current field."""
    # Create spatio-temporal field
    field = SpatioTemporalField(
        grid_size, time_steps, x_range, y_range, time_step_duration=3600.0
    )
    
    # Channel parameters
    channel_center_y = (y_range[1] - y_range[0]) / 2
    channel_width = (y_range[1] - y_range[0]) * 0.6
    
    # Narrow points in the channel
    narrow_points = [
        (x_range[0] + (x_range[1] - x_range[0]) * 0.3, channel_width * 0.7),
        (x_range[0] + (x_range[1] - x_range[0]) * 0.7, channel_width * 0.6)
    ]
    
    # Generate field for each time step
    for t in range(time_steps):
        # Create meshgrid
        x = np.linspace(x_range[0], x_range[1], grid_size[0])
        y = np.linspace(y_range[0], y_range[1], grid_size[1])
        X, Y = np.meshgrid(x, y)
        
        # Tidal factor changes with time
        time_hours = t * 3600.0 / 3600.0  # Convert to hours
        tidal_phase = 2 * np.pi * time_hours / tidal_period
        tidal_factor = np.sin(tidal_phase)  # -1 to +1
        
        # Initialize velocity components
        U = np.zeros(grid_size[::-1])
        V = np.zeros(grid_size[::-1])
        
        # Channel mask - points within the channel
        dist_from_center = np.abs(Y - channel_center_y)
        channel_mask = dist_from_center < (channel_width / 2)
        
        # Calculate velocity within channel
        max_speed = 1.5  # Max current speed in m/s
        
        for i in range(grid_size[1]):
            for j in range(grid_size[0]):
                if channel_mask[i, j]:
                    # Calculate local channel width
                    local_width = channel_width
                    for narrow_x, narrow_width in narrow_points:
                        dist_from_narrow = abs(X[i, j] - narrow_x)
                        if dist_from_narrow < 20:
                            weight = max(0, 1 - dist_from_narrow / 20)
                            local_width = local_width * (1 - weight) + narrow_width * weight
                    
                    # Parabolic velocity profile
                    y_rel = (Y[i, j] - channel_center_y) / (local_width / 2)
                    velocity_profile = 1 - (y_rel ** 2)
                    
                    # Width factor (faster in narrow parts)
                    width_factor = channel_width / local_width
                    
                    # Set along-channel velocity
                    U[i, j] = max_speed * tidal_factor * velocity_profile * width_factor
                    
                    # Add cross-channel components near narrow points
                    for narrow_x, _ in narrow_points:
                        dist_from_narrow = X[i, j] - narrow_x
                        if abs(dist_from_narrow) < 15:
                            if dist_from_narrow < 0:  # Before narrowing
                                V[i, j] = -0.1 * max_speed * np.sign(Y[i, j] - channel_center_y)
                            else:  # After narrowing
                                V[i, j] = 0.1 * max_speed * np.sign(Y[i, j] - channel_center_y)
        
        # Create a simple weather factor
        W = np.zeros(grid_size[::-1])
        for i in range(grid_size[1]):
            for j in range(grid_size[0]):
                W[i, j] = 0.3 + 0.2 * np.sin(X[i, j]/20 + tidal_phase)
        
        # Set field data for this time step
        field.set_field_at_time(t, U, V, W)
    
    return field


def create_channel_grid(field, grid_size):
    """Create a grid with channel and coastal features."""
    grid = TemporalNavigationGrid(
        grid_size=grid_size,
        time_steps=field.time_steps,
        cell_size=1.0
    )
    
    # Add the field to the grid
    grid.set_environment_field(field)
    
    # Channel parameters
    channel_center_y = grid_size[1] // 2
    channel_width = int(grid_size[1] * 0.6)
    
    # Add coast north and south of channel
    for x in range(grid_size[0]):
        # North shore
        north_start = min(grid_size[1]-1, channel_center_y + channel_width//2)
        variation_n = int(2 * np.sin(x / 8))
        for y in range(north_start + variation_n, grid_size[1]):
            if 0 <= y < grid_size[1]:
                grid.add_obstacle(x, y)
        
        # South shore
        south_end = max(0, channel_center_y - channel_width//2)
        variation_s = int(2 * np.cos(x / 10))
        for y in range(0, south_end - variation_s):
            if 0 <= y < grid_size[1]:
                grid.add_obstacle(x, y)
    
    # Add a few islands
    islands = [
        (grid_size[0]//5, channel_center_y - channel_width//6, 2),
        (grid_size[0]*3//4, channel_center_y + channel_width//5, 3)
    ]
    
    for ix, iy, ir in islands:
        for x in range(max(0, ix-ir), min(grid_size[0], ix+ir+1)):
            for y in range(max(0, iy-ir), min(grid_size[1], iy+ir+1)):
                if (x - ix)**2 + (y - iy)**2 <= ir**2:
                    if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
                        grid.add_obstacle(x, y)
    
    return grid


def plot_channel_at_time(grid, time_idx, filename=None):
    """Generate a plot of the channel at the specified time index."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot land (obstacles)
    land = np.zeros(grid.grid_size[::-1])
    for y in range(grid.grid_size[1]):
        for x in range(grid.grid_size[0]):
            if grid.is_obstacle(x, y):
                land[y, x] = 1
    
    ax.imshow(land, cmap='YlOrBr', alpha=0.7, extent=[0, grid.grid_size[0], 0, grid.grid_size[1]])
    
    # Plot current vectors
    quiver_density = 3
    X, Y = np.meshgrid(
        range(0, grid.grid_size[0], quiver_density),
        range(0, grid.grid_size[1], quiver_density)
    )
    
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    C = np.zeros_like(X, dtype=float)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            if not grid.is_obstacle(x, y):
                world_x, world_y = grid.cell_to_coords(x, y)
                u, v, _ = grid.environment_field.get_vector_at_position_time(world_x, world_y, time_idx)
                U[i, j] = u
                V[i, j] = v
                C[i, j] = np.sqrt(u**2 + v**2)
    
    # Mask points on land
    mask = (land[Y, X] == 0)
    quiver = ax.quiver(
        X[mask], Y[mask], 
        U[mask], V[mask], 
        C[mask],
        scale=20, cmap='viridis', 
        width=0.003, headwidth=4
    )
    
    # Add colorbar
    cbar = plt.colorbar(quiver, ax=ax)
    cbar.set_label('Current Speed (m/s)')
    
    # Add title and labels
    if time_idx < grid.time_steps / 2:
        tide_state = "Flood Tide"
    else:
        tide_state = "Ebb Tide"
    
    ax.set_title(f'Tidal Channel Flow at Time Index {time_idx} ({tide_state})')
    ax.set_xlabel('X Grid Cell')
    ax.set_ylabel('Y Grid Cell')
    ax.set_aspect('equal')
    
    # Save or show
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def generate_tidal_sequence(grid, output_dir="tidal_images"):
    """Generate a sequence of images showing the tidal cycle."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate images for key time points
    time_points = [0, 3, 6, 9]  # Quarter points in the tidal cycle
    
    for t in time_points:
        filename = os.path.join(output_dir, f"tidal_channel_t{t:02d}.png")
        print(f"Generating image for time {t}...")
        plot_channel_at_time(grid, t, filename)
    
    print(f"Generated {len(time_points)} images in {output_dir}/")


def generate_drift_analysis(grid, time_idx, output_dir="tidal_images"):
    """Generate a drift analysis visualization at the specified time."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot land (obstacles)
    land = np.zeros(grid.grid_size[::-1])
    for y in range(grid.grid_size[1]):
        for x in range(grid.grid_size[0]):
            if grid.is_obstacle(x, y):
                land[y, x] = 1
    
    ax.imshow(land, cmap='YlOrBr', alpha=0.7, extent=[0, grid.grid_size[0], 0, grid.grid_size[1]])
    
    # Plot current field
    quiver_density = 4
    X, Y = np.meshgrid(
        range(0, grid.grid_size[0], quiver_density),
        range(0, grid.grid_size[1], quiver_density)
    )
    
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    C = np.zeros_like(X, dtype=float)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            if not grid.is_obstacle(x, y):
                world_x, world_y = grid.cell_to_coords(x, y)
                u, v, _ = grid.environment_field.get_vector_at_position_time(world_x, world_y, time_idx)
                U[i, j] = u
                V[i, j] = v
                C[i, j] = np.sqrt(u**2 + v**2)
    
    # Mask points on land
    mask = (land[Y, X] == 0)
    quiver = ax.quiver(
        X[mask], Y[mask], 
        U[mask], V[mask], 
        C[mask],
        scale=25, cmap='Blues', alpha=0.7,
        width=0.002, headwidth=4
    )
    
    # Generate a set of drift start points
    drift_points = []
    stride = 8
    for x in range(stride, grid.grid_size[0]-stride, stride):
        for y in range(stride, grid.grid_size[1]-stride, stride):
            if not grid.is_obstacle(x, y):
                drift_points.append((x, y))
    
    # Calculate drift paths
    print(f"Calculating {len(drift_points)} drift paths...")
    drift_duration = 3.0  # hours
    
    for x, y in drift_points:
        # Convert to world coordinates
        world_x, world_y = grid.cell_to_coords(x, y)
        
        # Simulate drift
        drift_path = grid.environment_field.integrate_drift_path(
            world_x, world_y, time_idx, 
            duration_seconds=drift_duration * grid.time_step_duration
        )
        
        if drift_path and len(drift_path) > 1:
            # Convert back to cell coordinates
            drift_xs = []
            drift_ys = []
            
            for wx, wy, _ in drift_path:
                cell_x, cell_y = grid.coords_to_cell(wx, wy)
                drift_xs.append(cell_x)
                drift_ys.append(cell_y)
            
            # Calculate drift distance
            start_x, start_y = drift_xs[0], drift_ys[0]
            end_x, end_y = drift_xs[-1], drift_ys[-1]
            
            drift_dist = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            # Skip very short drift paths
            if drift_dist < 2:
                continue
            
            # Color by drift distance
            normalized_dist = min(1.0, drift_dist / 20)
            color = plt.cm.plasma(normalized_dist)
            
            # Plot the drift path
            ax.plot(drift_xs, drift_ys, '-', color=color, linewidth=1.5, alpha=0.8)
            
            # Mark start and end points
            ax.plot(start_x, start_y, 'o', color=color, markersize=3)
            ax.plot(end_x, end_y, 's', color=color, markersize=4)
    
    # Add colorbar for drift distance
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(0, 20))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Drift Distance (grid cells)')
    
    # Add title
    if time_idx < grid.time_steps / 2:
        tide_state = "Flood Tide"
    else:
        tide_state = "Ebb Tide"
        
    ax.set_title(f'Drift Analysis at Time {time_idx} ({tide_state})')
    
    # Save the figure
    filename = os.path.join(output_dir, f"drift_analysis_t{time_idx:02d}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved drift analysis to {filename}")
    plt.close()


def main():
    # Create tidal channel environment
    print("Creating tidal channel model...")
    grid_size = (60, 30)
    time_steps = 12
    field = create_tidal_channel(grid_size, time_steps)
    
    # Create navigation grid
    print("Creating navigation grid...")
    grid = create_channel_grid(field, grid_size)
    
    # Generate tidal sequence
    print("Generating tidal sequence...")
    generate_tidal_sequence(grid)
    
    # Generate drift analysis for flood and ebb tides
    print("Generating drift analysis...")
    generate_drift_analysis(grid, 3)  # Flood tide
    generate_drift_analysis(grid, 9)  # Ebb tide


if __name__ == "__main__":
    main()