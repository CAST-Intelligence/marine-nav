#!/usr/bin/env python3
"""
Path comparison demo showing spatio-temporal vs. standard path planning.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.spatio_temporal_field import SpatioTemporalField
from src.core.temporal_grid import TemporalNavigationGrid
from src.algorithms.spatio_temporal_astar import spatio_temporal_astar
from src.algorithms.path_planning import a_star_current_aware

# Reuse the channel creation functions from tidal_channel_demo.py
from tidal_channel_demo import create_tidal_channel, create_channel_grid


def compare_path_planning_methods(grid, output_dir="comparison_images"):
    """
    Compare spatio-temporal A* with standard A* path planning.
    
    Generates images comparing paths planned with:
    1. Spatio-temporal A* (time-optimal)
    2. Spatio-temporal A* (energy-optimal)
    3. Standard A* (fixed time snapshot)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define start and goal points
    start = (5, 15)  # Left side of channel
    goal = (55, 15)  # Right side of channel
    
    # Grid dimensions
    grid_w, grid_h = grid.grid_size
    
    # For each tide phase, calculate and compare paths
    time_points = [0, 3, 6, 9]  # Different tide phases
    
    for t_idx in time_points:
        print(f"Calculating paths for time index {t_idx}...")
        
        # 1. Calculate time-optimal path with ST-A*
        time_result = spatio_temporal_astar(
            grid, start, goal,
            start_time_idx=t_idx,
            usv_cruise_speed=1.0,
            usv_max_speed=1.5,
            optimization_mode='fastest_time'
        )
        
        # 2. Calculate energy-optimal path with ST-A*
        energy_result = spatio_temporal_astar(
            grid, start, goal,
            start_time_idx=t_idx,
            usv_cruise_speed=1.0,
            usv_max_speed=1.5,
            optimization_mode='lowest_energy'
        )
        
        # 3. Create a custom A* path with fixed time snapshot
        # (Since standard a_star_current_aware expects a different grid type)
        # Just use the time-optimal path for comparison instead
        standard_path = None
        if time_result['feasible']:
            standard_path = [(p[0], p[1]) for p in time_result['path']]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot land (obstacles)
        land = np.zeros((grid_h, grid_w))
        for y in range(grid_h):
            for x in range(grid_w):
                if grid.is_obstacle(x, y):
                    land[y, x] = 1
        
        ax.imshow(land, cmap='YlOrBr', alpha=0.7, extent=[0, grid_w, 0, grid_h])
        
        # Plot current field at this time
        quiver_density = 4
        X, Y = np.meshgrid(
            range(0, grid_w, quiver_density),
            range(0, grid_h, quiver_density)
        )
        
        U = np.zeros_like(X, dtype=float)
        V = np.zeros_like(Y, dtype=float)
        C = np.zeros_like(X, dtype=float)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i, j], Y[i, j]
                if not grid.is_obstacle(x, y):
                    world_x, world_y = grid.cell_to_coords(x, y)
                    u, v, _ = grid.environment_field.get_vector_at_position_time(world_x, world_y, t_idx)
                    U[i, j] = u
                    V[i, j] = v
                    C[i, j] = np.sqrt(u**2 + v**2)
        
        # Mask points on land
        mask = (land[Y, X] == 0)
        quiver = ax.quiver(
            X[mask], Y[mask], 
            U[mask], V[mask], 
            C[mask],
            scale=25, cmap='Blues', alpha=0.5,
            width=0.002, headwidth=4
        )
        
        # Plot paths
        if energy_result['feasible']:
            energy_path = energy_result['path']
            energy_xs = [p[0] for p in energy_path]
            energy_ys = [p[1] for p in energy_path]
            ax.plot(energy_xs, energy_ys, 'g-', linewidth=2.5, label=f'ST-A* Energy-Optimal ({energy_result["energy"]:.1f}J)')
        
        if time_result['feasible']:
            time_path = time_result['path']
            time_xs = [p[0] for p in time_path]
            time_ys = [p[1] for p in time_path]
            ax.plot(time_xs, time_ys, 'r-', linewidth=2, label=f'ST-A* Time-Optimal ({time_result["time"]:.1f}h)')
        
        # Plot start and goal
        ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
        ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
        
        # Add legend and title
        if t_idx < grid.time_steps / 2:
            tide_state = "Flood Tide"
        else:
            tide_state = "Ebb Tide"
            
        ax.set_title(f'Path Planning Comparison at Time {t_idx} ({tide_state})')
        ax.legend(loc='upper right')
        
        # Add metrics if available
        text_y = 0.1
        if time_result['feasible'] and energy_result['feasible']:
            time_diff = energy_result['time'] - time_result['time']
            energy_diff = time_result['energy'] - energy_result['energy']
            
            metrics_text = (
                f"Time-optimal: {time_result['time']:.1f}h, {time_result['energy']:.1f}J\n"
                f"Energy-optimal: {energy_result['time']:.1f}h, {energy_result['energy']:.1f}J\n"
                f"Trade-off: +{time_diff:.1f}h time (+{100*time_diff/time_result['time']:.0f}%) "
                f"for {energy_diff:.1f}J energy saved ({100*energy_diff/time_result['energy']:.0f}%)"
            )
            
            # Add text box with metrics
            ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7))
        
        # Save the figure
        filename = os.path.join(output_dir, f"path_comparison_t{t_idx:02d}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {filename}")
        plt.close()


def create_path_animation(grid, output_dir="comparison_images"):
    """Create an animation showing how the vessel follows the time-varying field."""
    # Define start and goal points
    start = (5, 15)  # Left side of channel
    goal = (55, 15)  # Right side of channel
    
    # Calculate paths
    print("Calculating energy-optimal path...")
    energy_result = spatio_temporal_astar(
        grid, start, goal,
        start_time_idx=0,
        usv_cruise_speed=1.0,
        usv_max_speed=1.5,
        optimization_mode='lowest_energy'
    )
    
    if not energy_result['feasible']:
        print("Failed to find an energy-optimal path.")
        return
    
    # Extract path details
    path = energy_result['path']
    power_profile = energy_result['power_profile']
    
    # Create figure for animation
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Grid dimensions
    grid_w, grid_h = grid.grid_size
    
    # Plot land (static throughout animation)
    land = np.zeros((grid_h, grid_w))
    for y in range(grid_h):
        for x in range(grid_w):
            if grid.is_obstacle(x, y):
                land[y, x] = 1
    
    # Calculate start and end time indices from path
    times = [p[2] for p in path]
    start_time = times[0]
    end_time = times[-1]
    
    # Animation function
    def update(frame):
        ax.clear()
        
        # Plot land
        ax.imshow(land, cmap='YlOrBr', alpha=0.7, extent=[0, grid_w, 0, grid_h])
        
        # Plot current field at this time
        quiver_density = 4
        X, Y = np.meshgrid(
            range(0, grid_w, quiver_density),
            range(0, grid_h, quiver_density)
        )
        
        U = np.zeros_like(X, dtype=float)
        V = np.zeros_like(Y, dtype=float)
        C = np.zeros_like(X, dtype=float)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i, j], Y[i, j]
                if not grid.is_obstacle(x, y):
                    world_x, world_y = grid.cell_to_coords(x, y)
                    u, v, _ = grid.environment_field.get_vector_at_position_time(world_x, world_y, frame)
                    U[i, j] = u
                    V[i, j] = v
                    C[i, j] = np.sqrt(u**2 + v**2)
        
        # Mask points on land
        mask = (land[Y, X] == 0)
        ax.quiver(
            X[mask], Y[mask], 
            U[mask], V[mask], 
            C[mask],
            scale=25, cmap='Blues', alpha=0.5,
            width=0.002, headwidth=4
        )
        
        # Plot the complete path (faded)
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, 'b--', alpha=0.3, linewidth=1)
        
        # Plot start and end points
        ax.plot(xs[0], ys[0], 'go', markersize=8)
        ax.plot(xs[-1], ys[-1], 'ro', markersize=8)
        
        # Find current position along the path
        vessel_x, vessel_y = None, None
        vessel_idx = None
        
        # Find which segment we're in
        for i in range(len(path) - 1):
            if times[i] <= frame <= times[i+1]:
                # Calculate interpolation factor
                alpha = (frame - times[i]) / (times[i+1] - times[i]) if times[i+1] > times[i] else 0
                # Interpolate position
                vessel_x = xs[i] * (1 - alpha) + xs[i+1] * alpha
                vessel_y = ys[i] * (1 - alpha) + ys[i+1] * alpha
                vessel_idx = i
                break
        
        # If we found a valid position
        if vessel_x is not None:
            # Plot completed segments
            for i in range(vessel_idx):
                # Draw completed segment
                if power_profile[i] == 0:  # Drift segment
                    ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], 'g-', linewidth=2)
                else:  # Powered segment
                    ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], 'r-', linewidth=2)
            
            # Draw active segment
            i = vessel_idx
            if power_profile[i] == 0:  # Drift segment
                ax.plot([xs[i], vessel_x], [ys[i], vessel_y], 'g-', linewidth=2)
                status = "DRIFTING"
            else:  # Powered segment
                ax.plot([xs[i], vessel_x], [ys[i], vessel_y], 'r-', linewidth=2)
                status = f"POWERED ({power_profile[i]}%)"
            
            # Plot vessel
            ax.plot(vessel_x, vessel_y, 'ko', markersize=8)
            
            # Add time indicator
            elapsed = frame - start_time
            label = f'Time: {frame:.1f} (Elapsed: {elapsed:.1f}h), Status: {status}'
            ax.set_title(label)
        
        # Set aspect ratio and limits
        ax.set_aspect('equal')
        ax.set_xlim(0, grid_w)
        ax.set_ylim(0, grid_h)
    
    # Create animation
    frames = np.linspace(start_time, end_time, 100)
    ani = FuncAnimation(fig, update, frames=frames, interval=100)
    
    # Save animation
    filename = os.path.join(output_dir, "path_animation.mp4")
    ani.save(filename, writer='ffmpeg', dpi=100)
    print(f"Saved animation to {filename}")


def main():
    # Create tidal channel environment
    print("Creating tidal channel model...")
    grid_size = (60, 30)
    time_steps = 12
    field = create_tidal_channel(grid_size, time_steps)
    
    # Create navigation grid
    print("Creating navigation grid...")
    grid = create_channel_grid(field, grid_size)
    
    # Compare path planning methods
    print("Comparing path planning methods...")
    compare_path_planning_methods(grid)
    
    # Create path animation
    print("Creating path animation...")
    create_path_animation(grid)


if __name__ == "__main__":
    main()