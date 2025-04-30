#!/usr/bin/env python3
"""
Demonstration of the spatio-temporal path planning algorithm.

This example shows how to:
1. Create a time-varying environment with currents
2. Build a temporal navigation grid
3. Run the ST-A* algorithm for both time and energy optimization
4. Visualize the results, including drift opportunities
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.spatio_temporal_field import SpatioTemporalField
from src.core.temporal_grid import TemporalNavigationGrid
from src.algorithms.spatio_temporal_astar import (
    spatio_temporal_astar, 
    analyze_drift_opportunities
)


def create_time_varying_current(
    grid_size=(50, 50), 
    time_steps=24,
    x_range=(0, 100),
    y_range=(0, 100),
    time_step_duration=3600.0
):
    """
    Create a time-varying current field with a rotating gyre pattern
    that evolves over time.
    """
    # Create spatio-temporal field
    field = SpatioTemporalField(
        grid_size, time_steps, x_range, y_range, time_step_duration
    )
    
    # Generate field data for each time step
    center_x = (x_range[1] - x_range[0]) / 2
    center_y = (y_range[1] - y_range[0]) / 2
    
    for t in range(time_steps):
        # Create meshgrid for this time step
        x = np.linspace(x_range[0], x_range[1], grid_size[0])
        y = np.linspace(y_range[0], y_range[1], grid_size[1])
        X, Y = np.meshgrid(x, y)
        
        # Time-varying factors
        phase = 2 * np.pi * t / time_steps
        strength = 0.5 + 0.3 * np.sin(phase)
        
        # Gyre pattern with time variation
        dx = X - center_x
        dy = Y - center_y
        distance = np.sqrt(dx**2 + dy**2)
        mask = (distance != 0)
        
        # Initialize U and V
        U = np.zeros(grid_size[::-1])
        V = np.zeros(grid_size[::-1])
        
        # Current speed peaks at distance = 20 from center
        radial_factor = np.exp(-((distance - 20) / 15)**2)
        
        # Tangential velocity (creates circular flow)
        U[mask] = -strength * radial_factor[mask] * dy[mask] / distance[mask]
        V[mask] = strength * radial_factor[mask] * dx[mask] / distance[mask]
        
        # Add a general flow in time-varying direction
        base_angle = phase
        base_u = 0.2 * np.cos(base_angle) 
        base_v = 0.2 * np.sin(base_angle)
        
        U += base_u
        V += base_v
        
        # Create weather factor (higher in center, varies with time)
        W = 0.2 + 0.8 * np.exp(-distance / 30) * (0.5 + 0.5 * np.sin(phase))
        
        # Set field data for this time step
        field.set_field_at_time(t, U, V, W)
    
    return field


def create_obstacle_pattern(grid, pattern="islands"):
    """Create a pattern of obstacles in the grid."""
    
    if pattern == "islands":
        # Create island obstacles
        # Central island
        center_x = grid.grid_size[0] // 2
        center_y = grid.grid_size[1] // 2
        radius = min(grid.grid_size) // 6
        
        for x in range(grid.grid_size[0]):
            for y in range(grid.grid_size[1]):
                # Distance from center
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < radius:
                    grid.add_obstacle(x, y)
        
        # Add a few smaller islands
        islands = [
            (center_x + radius*2, center_y + radius*2, radius//2),
            (center_x - radius*2, center_y - radius*2, radius//2),
            (center_x + radius*2, center_y - radius*2, radius//3),
        ]
        
        for ix, iy, ir in islands:
            for x in range(max(0, ix-ir), min(grid.grid_size[0], ix+ir+1)):
                for y in range(max(0, iy-ir), min(grid.grid_size[1], iy+ir+1)):
                    dist = np.sqrt((x - ix)**2 + (y - iy)**2)
                    if dist < ir:
                        grid.add_obstacle(x, y)
    
    elif pattern == "channel":
        # Create a narrow channel with islands
        channel_width = grid.grid_size[1] // 5
        channel_y = grid.grid_size[1] // 2
        
        # Add obstacles above and below the channel
        for x in range(grid.grid_size[0]):
            for y in range(grid.grid_size[1]):
                if y < channel_y - channel_width//2 or y > channel_y + channel_width//2:
                    grid.add_obstacle(x, y)
        
        # Add some island obstacles in the channel
        islands = [
            (grid.grid_size[0]//4, channel_y, channel_width//3),
            (grid.grid_size[0]*3//4, channel_y, channel_width//3),
        ]
        
        for ix, iy, ir in islands:
            for x in range(max(0, ix-ir), min(grid.grid_size[0], ix+ir+1)):
                for y in range(max(0, iy-ir), min(grid.grid_size[1], iy+ir+1)):
                    dist = np.sqrt((x - ix)**2 + (y - iy)**2)
                    if dist < ir:
                        grid.add_obstacle(x, y)
    
    return grid


def plot_environment_at_time(ax, grid, time_idx, quiver_scale=30, quiver_density=2):
    """Plot the environment state at a specific time index."""
    # Clear the axes
    ax.clear()
    
    # Plot obstacles
    obstacle_map = np.zeros(grid.grid_size[::-1])
    for y in range(grid.grid_size[1]):
        for x in range(grid.grid_size[0]):
            if grid.is_obstacle(x, y):
                obstacle_map[y, x] = 1
    
    ax.imshow(obstacle_map, cmap='gray', alpha=0.6, interpolation='nearest', extent=[0, grid.grid_size[0], 0, grid.grid_size[1]])
    
    # Plot current vectors using quiver
    if grid.environment_field:
        X, Y = np.meshgrid(range(0, grid.grid_size[0], quiver_density), 
                          range(0, grid.grid_size[1], quiver_density))
        U = np.zeros_like(X, dtype=float)
        V = np.zeros_like(Y, dtype=float)
        C = np.zeros_like(X, dtype=float)  # For coloring by magnitude
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i, j], Y[i, j]
                world_x, world_y = grid.cell_to_coords(x, y)
                u, v, w = grid.environment_field.get_vector_at_position_time(world_x, world_y, time_idx)
                U[i, j] = u
                V[i, j] = v
                C[i, j] = np.sqrt(u**2 + v**2)  # Magnitude
        
        quiver = ax.quiver(X, Y, U, V, C, scale=quiver_scale, cmap='viridis', width=0.002, headwidth=4, headlength=5)
        
        # Add a colorbar
        cbar = plt.colorbar(quiver, ax=ax)
        cbar.set_label('Current Magnitude (m/s)')
    
    # Set title and labels
    ax.set_title(f'Environment at Time Index: {time_idx:.2f}')
    ax.set_xlabel('X Grid Cell')
    ax.set_ylabel('Y Grid Cell')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    return ax


def visualize_path(grid, path_result, drift_segments=None, animation_file=None):
    """
    Visualize the path and environment, with option to create animation.
    
    Args:
        grid: The temporal navigation grid
        path_result: Result dictionary from spatio_temporal_astar
        drift_segments: List of segment indices that are drift opportunities
        animation_file: If provided, save animation to this file
    """
    path = path_result['path']
    power_profile = path_result['power_profile']
    
    if not path:
        print("No path to visualize")
        return
    
    # Extract coordinates and time indices
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    times = [p[2] for p in path]
    
    # Determine time range for visualization
    min_time = min(times)
    max_time = max(times)
    time_steps = int(max_time - min_time) + 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # If creating animation
    if animation_file:
        frames = []
        # Create frames for each time step
        for t_idx in np.linspace(min_time, max_time, 100):
            # Plot environment at this time
            plot_environment_at_time(ax, grid, t_idx)
            
            # Plot full path (faded)
            ax.plot(xs, ys, 'r--', alpha=0.3, label='Full Path')
            
            # Find path segments that are active at this time
            active_segments = []
            for i in range(len(path) - 1):
                if times[i] <= t_idx <= times[i+1]:
                    active_segments.append(i)
            
            # Plot active segments
            for i in active_segments:
                # Interpolate position at this time
                if times[i+1] > times[i]:
                    alpha = (t_idx - times[i]) / (times[i+1] - times[i])
                    x = xs[i] * (1 - alpha) + xs[i+1] * alpha
                    y = ys[i] * (1 - alpha) + ys[i+1] * alpha
                    
                    # Is this a drift segment?
                    if drift_segments and i in drift_segments:
                        ax.plot([xs[i], x], [ys[i], y], 'b-', linewidth=3, label='Drift' if i == active_segments[0] else '')
                    else:
                        ax.plot([xs[i], x], [ys[i], y], 'g-', linewidth=3, label='Powered' if i == active_segments[0] else '')
                    
                    # Plot current position
                    ax.plot(x, y, 'ro', markersize=10)
            
            # Add time indicator
            ax.set_title(f'Time: {t_idx:.2f}')
            
            # Save frame
            frames.append([plt.gcf()])
        
        # Create animation
        ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
        if animation_file:
            ani.save(animation_file)
    
    # Create static plot
    else:
        # Plot environment at start time
        plot_environment_at_time(ax, grid, min_time)
        
        # Plot path
        ax.plot(xs, ys, 'r-', label='Path')
        
        # Highlight start and end
        ax.plot(xs[0], ys[0], 'go', markersize=12, label='Start')
        ax.plot(xs[-1], ys[-1], 'ro', markersize=12, label='Goal')
        
        # Highlight drift segments
        if drift_segments:
            for i in drift_segments:
                ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], 'b-', linewidth=4, label='Drift' if i == drift_segments[0] else '')
        
        # Add legend
        ax.legend()
    
    # Show results
    plt.tight_layout()
    
    # Print summary
    print(f"Path Summary:")
    print(f"  - Total Time: {path_result['time']:.2f}")
    print(f"  - Total Energy: {path_result['energy']:.2f}")
    print(f"  - Path Length: {len(path)}")
    if drift_segments:
        print(f"  - Drift Segments: {len(drift_segments)}")
    
    plt.show()


def main():
    # Create a time-varying current field
    print("Creating time-varying current field...")
    grid_size = (50, 50)
    time_steps = 24
    field = create_time_varying_current(
        grid_size=grid_size, 
        time_steps=time_steps,
        x_range=(0, 100),
        y_range=(0, 100)
    )
    
    # Create temporal navigation grid
    print("Creating navigation grid...")
    grid = TemporalNavigationGrid(
        grid_size=grid_size,
        time_steps=time_steps,
        cell_size=2.0
    )
    
    # Add the current field to the grid
    grid.set_environment_field(field)
    
    # Add obstacles
    print("Adding obstacles...")
    create_obstacle_pattern(grid, pattern="islands")
    
    # Define start and end points
    start = (5, 5)
    goal = (45, 45)
    
    # Run ST-A* for fastest time
    print("\nFinding fastest time path...")
    time_result = spatio_temporal_astar(
        grid, start, goal,
        start_time_idx=0.0,
        usv_cruise_speed=1.0,
        usv_max_speed=1.5,
        optimization_mode='fastest_time'
    )
    
    if time_result['feasible']:
        print(f"Fastest time path found! Time: {time_result['time']:.2f}, Energy: {time_result['energy']:.2f}")
        # Analyze for drift (should be few or none in time-optimal path)
        time_drift_segments = analyze_drift_opportunities(grid, time_result['path'], time_result['power_profile'])
        
        # Visualize
        visualize_path(grid, time_result, time_drift_segments)
    else:
        print("Could not find a fastest time path")
    
    # Run ST-A* for lowest energy
    print("\nFinding lowest energy path...")
    energy_result = spatio_temporal_astar(
        grid, start, goal,
        start_time_idx=0.0,
        usv_cruise_speed=1.0,
        usv_max_speed=1.5,
        optimization_mode='lowest_energy'
    )
    
    if energy_result['feasible']:
        print(f"Lowest energy path found! Time: {energy_result['time']:.2f}, Energy: {energy_result['energy']:.2f}")
        # Analyze for drift (should find many in energy-optimal path)
        energy_drift_segments = analyze_drift_opportunities(grid, energy_result['path'], energy_result['power_profile'])
        
        # Visualize
        visualize_path(grid, energy_result, energy_drift_segments)
        
        # Create animation (optional)
        # visualize_path(grid, energy_result, energy_drift_segments, animation_file="energy_path_animation.mp4")
    else:
        print("Could not find a lowest energy path")
    
    # Print comparison
    if time_result['feasible'] and energy_result['feasible']:
        time_diff = energy_result['time'] - time_result['time']
        energy_diff = time_result['energy'] - energy_result['energy']
        
        print("\nComparison:")
        print(f"  - Time Difference: +{time_diff:.2f} ({time_diff/time_result['time']*100:.1f}% longer for energy-optimal)")
        print(f"  - Energy Difference: -{energy_diff:.2f} ({energy_diff/time_result['energy']*100:.1f}% saved in energy-optimal)")
        
        if len(time_drift_segments) > 0:
            print(f"  - Time-optimal path has {len(time_drift_segments)} drift segments")
        else:
            print(f"  - Time-optimal path has no drift segments")
        
        print(f"  - Energy-optimal path has {len(energy_drift_segments)} drift segments")


if __name__ == "__main__":
    main()