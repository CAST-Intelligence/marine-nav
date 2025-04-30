#!/usr/bin/env python3
"""
Demonstration of realistic environmental patterns for marine navigation.

This demo shows:
1. Tidal flow in a channel (reversing with time)
2. Wind patterns overlaid on the water currents
3. Animations of these time-varying fields
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.spatio_temporal_field import SpatioTemporalField
from src.core.temporal_grid import TemporalNavigationGrid


def create_tidal_channel(
    grid_size=(100, 50), 
    time_steps=24,
    x_range=(0, 100),
    y_range=(0, 50),
    time_step_duration=3600.0,  # 1 hour in seconds
    tidal_period=12.42,         # Tidal period in hours (average M2 tide)
    max_current_speed=1.5       # Max current speed in m/s
):
    """
    Create a time-varying current field simulating tidal flow in a channel.
    
    The tide flows in and out of the channel, with velocity proportional to
    position along the channel. Current is stronger in the center of the channel.
    """
    # Create spatio-temporal field
    field = SpatioTemporalField(
        grid_size, time_steps, x_range, y_range, time_step_duration
    )
    
    # Channel parameters
    channel_center_y = (y_range[1] - y_range[0]) / 2
    channel_width = (y_range[1] - y_range[0]) * 0.6  # 60% of height
    
    # Narrow points in the channel
    narrow_points = [
        (x_range[0] + (x_range[1] - x_range[0]) * 0.3, channel_width * 0.7),  # 30% along, 70% width
        (x_range[0] + (x_range[1] - x_range[0]) * 0.7, channel_width * 0.6)   # 70% along, 60% width
    ]
    
    # Generate field data for each time step
    for t in range(time_steps):
        # Create meshgrid for this time step
        x = np.linspace(x_range[0], x_range[1], grid_size[0])
        y = np.linspace(y_range[0], y_range[1], grid_size[1])
        X, Y = np.meshgrid(x, y)
        
        # Time-varying tidal factor 
        # Sine wave with tidal_period (hours)
        time_hours = t * time_step_duration / 3600.0  # Convert to hours
        tidal_phase = 2 * np.pi * time_hours / tidal_period
        tidal_factor = np.sin(tidal_phase)  # -1 to +1 (ebb to flood)
        
        # Initialize U (east-west) and V (north-south) components
        U = np.zeros(grid_size[::-1])
        V = np.zeros(grid_size[::-1])
        
        # Calculate distance from channel center for each point
        dist_from_center = np.abs(Y - channel_center_y)
        
        # Create channel mask - True for points within the channel
        channel_mask = dist_from_center < (channel_width / 2)
        
        # Calculate base U (along-channel) velocity
        # Velocity profile is parabolic across channel (maximum in center)
        for i in range(grid_size[1]):  # For each y-row
            for j in range(grid_size[0]):  # For each x-column
                if channel_mask[i, j]:
                    # Calculate local channel width (narrower at narrow points)
                    local_width = channel_width
                    for narrow_x, narrow_width in narrow_points:
                        # Weight based on distance from narrow point
                        dist_from_narrow = abs(X[i, j] - narrow_x)
                        if dist_from_narrow < 20:  # Influence zone
                            weight = max(0, 1 - dist_from_narrow / 20) 
                            local_width = local_width * (1 - weight) + narrow_width * weight
                    
                    # Parabolic profile: 0 at edges, 1 at center
                    y_rel = (Y[i, j] - channel_center_y) / (local_width / 2)
                    velocity_profile = 1 - (y_rel ** 2)
                    
                    # Higher velocity in narrow sections (conservation of mass)
                    width_factor = channel_width / local_width
                    
                    # Set U component - along channel flow that changes with tide
                    U[i, j] = max_current_speed * tidal_factor * velocity_profile * width_factor
                    
                    # Small cross-channel flow in bends
                    # Stronger at narrowing points
                    for narrow_x, _ in narrow_points:
                        dist_from_narrow = X[i, j] - narrow_x
                        if abs(dist_from_narrow) < 15:  # Near the narrowing
                            # Cross flow towards centerline before narrowing, away after
                            if dist_from_narrow < 0:  # Before narrowing
                                V[i, j] = -0.1 * max_current_speed * np.sign(Y[i, j] - channel_center_y)
                            else:  # After narrowing
                                V[i, j] = 0.1 * max_current_speed * np.sign(Y[i, j] - channel_center_y)
        
        # Create weather factor (wind) - more complex pattern
        # Wind has its own independent pattern changing with time
        W = np.zeros(grid_size[::-1])
        
        # Wind changes direction throughout the day
        wind_phase = 2 * np.pi * time_hours / 24.0  # Full cycle each day
        wind_dir_x = np.cos(wind_phase)
        wind_dir_y = np.sin(wind_phase)
        
        # Base wind field with some spatial variation
        for i in range(grid_size[1]):
            for j in range(grid_size[0]):
                # Distance-based variation
                dist_factor = 0.5 + 0.5 * np.sin(X[i, j] / 30) * np.cos(Y[i, j] / 20)
                # Time variation
                time_factor = 0.5 + 0.5 * np.sin(tidal_phase + X[i, j] / 50)
                
                # Wind strength (0-1 scale)
                W[i, j] = dist_factor * time_factor
        
        # Set field data for this time step
        field.set_field_at_time(t, U, V, W)
    
    return field


def create_navigation_grid(field, grid_size=(100, 50)):
    """Create a navigation grid with channel and coastal obstacles."""
    
    # Create temporal navigation grid matching the field
    grid = TemporalNavigationGrid(
        grid_size=grid_size,
        time_steps=field.time_steps,
        cell_size=1.0
    )
    
    # Add the current field to the grid
    grid.set_environment_field(field)
    
    # Channel parameters
    channel_center_y = grid_size[1] // 2
    channel_width = int(grid_size[1] * 0.6)  # 60% of height
    
    # Add land obstacles (north and south of channel)
    for x in range(grid_size[0]):
        if x >= grid_size[0]:  # Safety check
            continue
            
        # North shore
        north_start = min(grid_size[1]-1, channel_center_y + channel_width//2)
        for y in range(north_start, grid_size[1]):
            # Add some coastline variation
            variation = int(2 * np.sin(x / 10)) 
            variation_pos = min(grid_size[1]-1, channel_center_y + channel_width//2 + variation)
            if y >= variation_pos:
                try:
                    grid.add_obstacle(x, y)
                except ValueError:
                    # Skip if out of bounds
                    pass
        
        # South shore
        south_end = max(0, channel_center_y - channel_width//2)
        for y in range(0, south_end):
            # Add some coastline variation
            variation = int(2 * np.cos(x / 12))
            variation_pos = max(0, channel_center_y - channel_width//2 - variation)
            if y <= variation_pos:
                try:
                    grid.add_obstacle(x, y)
                except ValueError:
                    # Skip if out of bounds
                    pass
    
    # Add some islands in the channel
    islands = [
        (grid_size[0]//5, channel_center_y + channel_width//6, 3),  # Small island near north shore
        (grid_size[0]//2, channel_center_y, 2),                     # Tiny island in center
        (grid_size[0]*3//4, channel_center_y - channel_width//5, 4) # Medium island near south shore
    ]
    
    for ix, iy, ir in islands:
        for x in range(max(0, ix-ir), min(grid_size[0], ix+ir+1)):
            for y in range(max(0, iy-ir), min(grid_size[1], iy+ir+1)):
                dist = np.sqrt((x - ix)**2 + (y - iy)**2)
                if dist < ir:
                    try:
                        grid.add_obstacle(x, y)
                    except ValueError:
                        # Skip if out of bounds
                        pass
    
    return grid


def plot_environment_state(ax, grid, time_idx, plot_type='current'):
    """
    Plot the environment state at a specific time index.
    
    Args:
        ax: Matplotlib axis
        grid: Navigation grid
        time_idx: Time index to plot
        plot_type: 'current', 'wind', or 'combined'
    """
    # Clear the axes
    ax.clear()
    
    # Plot obstacles
    obstacle_map = np.zeros(grid.grid_size[::-1])
    for y in range(grid.grid_size[1]):
        for x in range(grid.grid_size[0]):
            if grid.is_obstacle(x, y):
                obstacle_map[y, x] = 1
    
    # Plot land masses in tan color
    land = ax.imshow(obstacle_map, cmap='YlOrBr', alpha=0.8, 
                    interpolation='nearest', 
                    extent=[0, grid.grid_size[0], 0, grid.grid_size[1]])
    
    # Set up grid for current/wind vectors
    quiver_density = 5  # Space between vectors
    X, Y = np.meshgrid(
        range(0, grid.grid_size[0], quiver_density), 
        range(0, grid.grid_size[1], quiver_density)
    )
    
    # Initialize arrays for plotting
    U_current = np.zeros_like(X, dtype=float)  # Current x-component
    V_current = np.zeros_like(Y, dtype=float)  # Current y-component
    U_wind = np.zeros_like(X, dtype=float)     # Wind x-component 
    V_wind = np.zeros_like(Y, dtype=float)     # Wind y-component
    C_current = np.zeros_like(X, dtype=float)  # Current magnitude
    W_factor = np.zeros_like(X, dtype=float)   # Wind factor
    
    # Extract field data for plotting
    if grid.environment_field:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i, j], Y[i, j]
                
                # Skip points on land
                if grid.is_obstacle(x, y):
                    continue
                    
                # Get vector components and weather at this position and time
                world_x, world_y = grid.cell_to_coords(x, y)
                u, v, w = grid.environment_field.get_vector_at_position_time(
                    world_x, world_y, time_idx
                )
                
                # Store values for plotting
                U_current[i, j] = u
                V_current[i, j] = v
                C_current[i, j] = np.sqrt(u**2 + v**2)  # Current magnitude
                W_factor[i, j] = w
                
                # For wind visualization, generate wind vectors from weather factor
                # Here we're simulating the wind direction based on the weather factor
                # and some arbitrary function of position and time
                wind_phase = 2 * np.pi * time_idx / 24  # Full cycle each day
                wind_dir_x = np.cos(wind_phase + x/20)
                wind_dir_y = np.sin(wind_phase + y/20)
                wind_magnitude = w  # Use weather factor as wind magnitude
                
                U_wind[i, j] = wind_dir_x * wind_magnitude
                V_wind[i, j] = wind_dir_y * wind_magnitude
    
    # Determine what to plot based on plot_type
    if plot_type == 'current' or plot_type == 'combined':
        # Plot current vectors
        current_quiver = ax.quiver(
            X, Y, U_current, V_current, C_current, 
            scale=25, cmap='viridis', 
            width=0.002, headwidth=4, headlength=5
        )
        
        # Add colorbar for current
        current_cbar = plt.colorbar(current_quiver, ax=ax)
        current_cbar.set_label('Current Magnitude (m/s)')
    
    if plot_type == 'wind' or plot_type == 'combined':
        # For wind, we'll use a different color and plot on top
        # Only plot wind where it's significant
        wind_mask = W_factor > 0.3
        if np.any(wind_mask):
            # For combined plot, make wind less prominent
            if plot_type == 'combined':
                alpha = 0.6
                wind_scale = 40
            else:
                alpha = 0.9
                wind_scale = 25
                
            # Plot wind vectors
            wind_quiver = ax.quiver(
                X[wind_mask], Y[wind_mask], 
                U_wind[wind_mask], V_wind[wind_mask],
                W_factor[wind_mask],
                scale=wind_scale, cmap='autumn', 
                width=0.001, headwidth=3, headlength=4,
                alpha=alpha
            )
            
            # Add colorbar for wind
            if plot_type == 'wind':
                wind_cbar = plt.colorbar(wind_quiver, ax=ax)
                wind_cbar.set_label('Wind Strength')
    
    # Set title and labels
    title = f'Environment at Time: {time_idx:.1f}'
    if plot_type == 'current':
        title += ' (Current Only)'
    elif plot_type == 'wind':
        title += ' (Wind Only)'
    elif plot_type == 'combined':
        title += ' (Current & Wind)'
        
    ax.set_title(title)
    ax.set_xlabel('X Grid Cell')
    ax.set_ylabel('Y Grid Cell')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return ax


def create_environment_animation(grid, output_file=None, plot_type='combined'):
    """
    Create an animation of the time-varying environment.
    
    Args:
        grid: Temporal navigation grid
        output_file: If provided, save animation to this file
        plot_type: 'current', 'wind', or 'combined'
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Determine time steps to animate
    time_steps = np.linspace(0, grid.time_steps-1, 100)
    
    # Create animation frames
    frames = []
    print(f"Creating {len(time_steps)} animation frames...")
    
    for t_idx in time_steps:
        # Clear previous frame
        plt.clf()
        ax = plt.gca()
        
        # Plot environment at this time step
        plot_environment_state(ax, grid, t_idx, plot_type)
        plt.tight_layout()
        
        # Save frame
        frames.append([plt.gcf()])
    
    # Create animation
    print("Generating animation...")
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=False)
    
    # Save if output file provided
    if output_file:
        print(f"Saving animation to {output_file}...")
        ani.save(output_file, writer='ffmpeg')
    
    plt.tight_layout()
    plt.show()


def plot_path_comparison(grid, plot_title="Path Comparison"):
    """Create a comparative visualization for path planning with different tide states."""
    from src.algorithms.spatio_temporal_astar import spatio_temporal_astar

    # Fixed starting and ending points (across the channel)
    start = (10, 15)  # Left side
    goal = (90, 35)   # Right side
    
    # Plan paths at different tide phases
    tide_phases = [0, 6, 12, 18]  # 0, 6, 12, 18 hours (covering full tidal cycle)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    paths = []
    
    # For each tide phase
    for i, time_idx in enumerate(tide_phases):
        ax = axes[i]
        
        # Plot the environment at this time
        plot_environment_state(ax, grid, time_idx, 'current')
        
        # Find both time-optimal and energy-optimal paths
        time_result = spatio_temporal_astar(
            grid, start, goal,
            start_time_idx=time_idx,
            usv_cruise_speed=1.0,
            usv_max_speed=1.5,
            optimization_mode='fastest_time'
        )
        
        energy_result = spatio_temporal_astar(
            grid, start, goal,
            start_time_idx=time_idx,
            usv_cruise_speed=1.0,
            usv_max_speed=1.5,
            optimization_mode='lowest_energy'
        )
        
        paths.append((time_result, energy_result))
        
        # Plot the paths if found
        if time_result['feasible']:
            time_path = time_result['path']
            time_xs = [p[0] for p in time_path]
            time_ys = [p[1] for p in time_path]
            ax.plot(time_xs, time_ys, 'r-', linewidth=2, label='Time-optimal')
            ax.plot(time_xs[0], time_ys[0], 'go', markersize=8)  # Start
            ax.plot(time_xs[-1], time_ys[-1], 'ro', markersize=8)  # End
        
        if energy_result['feasible']:
            energy_path = energy_result['path']
            energy_xs = [p[0] for p in energy_path]
            energy_ys = [p[1] for p in energy_path]
            ax.plot(energy_xs, energy_ys, 'b-', linewidth=2, label='Energy-optimal')
            
        # Add legend and title
        ax.legend()
        phase_name = {0: "Flood Start", 6: "High Tide", 12: "Ebb Start", 18: "Low Tide"}
        ax.set_title(f"Paths at {phase_name.get(time_idx, f'Time {time_idx}')} (t={time_idx})")
        
    # Add overall title
    fig.suptitle(plot_title, fontsize=16)
    plt.tight_layout()
    
    # Print comparison
    print("\nPath Comparison at Different Tide Phases:")
    for i, time_idx in enumerate(tide_phases):
        time_result, energy_result = paths[i]
        print(f"\nTime Index {time_idx}:")
        
        if time_result['feasible'] and energy_result['feasible']:
            time_diff = energy_result['time'] - time_result['time']
            energy_diff = time_result['energy'] - energy_result['energy']
            
            print(f"  - Time-optimal path: Time={time_result['time']:.2f}, Energy={time_result['energy']:.2f}")
            print(f"  - Energy-optimal path: Time={energy_result['time']:.2f}, Energy={energy_result['energy']:.2f}")
            print(f"  - Tradeoff: +{time_diff:.2f} time (+{100*time_diff/time_result['time']:.1f}%) for {energy_diff:.2f} energy saved ({100*energy_diff/time_result['energy']:.1f}%)")
        else:
            if not time_result['feasible']:
                print("  - No time-optimal path found")
            if not energy_result['feasible']:
                print("  - No energy-optimal path found")
    
    plt.show()
    
    return paths


def create_path_animation(grid, path_result, drift_segments=None, output_file=None):
    """
    Create an animation showing a vessel following the computed path through the time-varying environment.
    
    Args:
        grid: Temporal navigation grid
        path_result: Result dictionary from spatio_temporal_astar
        drift_segments: List of indices indicating drift segments
        output_file: If provided, save animation to this file
    """
    from src.algorithms.spatio_temporal_astar import analyze_drift_opportunities
    
    path = path_result['path']
    power_profile = path_result['power_profile']
    
    if not drift_segments and path and power_profile:
        # Analyze drift opportunities if not provided
        drift_segments = analyze_drift_opportunities(grid, path, power_profile)
    
    if not path:
        print("No path to animate")
        return
    
    # Extract coordinates and time indices
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    times = [p[2] for p in path]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Determine time steps for animation frames
    # We want enough frames to make smooth animation
    num_frames = 100
    path_duration = times[-1] - times[0]
    time_steps = np.linspace(times[0], times[-1], num_frames)
    
    # Record frames
    frames = []
    vessel_path_x = []
    vessel_path_y = []
    
    # Create animation frames
    for frame_idx, t_idx in enumerate(time_steps):
        # Plot environment at this time
        plot_environment_state(ax, grid, t_idx, 'current')
        
        # Plot the complete path (faded)
        ax.plot(xs, ys, 'r--', alpha=0.3, label='Complete Path')
        
        # Find all segments that are active or completed at this time
        active_segment = None
        for i in range(len(path) - 1):
            if times[i] <= t_idx <= times[i+1]:
                active_segment = i
                break
        
        # Plot completed segments
        for i in range(len(path) - 1):
            if times[i+1] <= t_idx:  # Segment is completed
                # Determine if this is a drift segment
                is_drift = drift_segments and i in drift_segments
                
                # Plot the segment with appropriate style
                if is_drift:
                    ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], 'b-', linewidth=3)
                else:
                    ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], 'g-', linewidth=3)
        
        # Plot active segment and vessel
        if active_segment is not None:
            i = active_segment
            start_time = times[i]
            end_time = times[i+1]
            
            # Calculate interpolation factor
            if end_time > start_time:  # Avoid division by zero
                alpha = (t_idx - start_time) / (end_time - start_time)
                # Interpolate position
                vessel_x = xs[i] * (1 - alpha) + xs[i+1] * alpha
                vessel_y = ys[i] * (1 - alpha) + ys[i+1] * alpha
                
                # Draw active segment
                is_drift = drift_segments and i in drift_segments
                if is_drift:
                    ax.plot([xs[i], vessel_x], [ys[i], vessel_y], 'b-', linewidth=3, label='Drift' if i == 0 or frame_idx == 0 else "")
                else:
                    ax.plot([xs[i], vessel_x], [ys[i], vessel_y], 'g-', linewidth=3, label='Powered' if i == 0 or frame_idx == 0 else "")
                
                # Add vessel position to path
                vessel_path_x.append(vessel_x)
                vessel_path_y.append(vessel_y)
                
                # Plot vessel path (wake)
                if len(vessel_path_x) > 1:
                    # Fade the wake based on age
                    wake_length = 20  # How many points to show in wake
                    if len(vessel_path_x) > wake_length:
                        wake_x = vessel_path_x[-wake_length:]
                        wake_y = vessel_path_y[-wake_length:]
                    else:
                        wake_x = vessel_path_x
                        wake_y = vessel_path_y
                    
                    # Draw wake with gradient alpha
                    for j in range(len(wake_x) - 1):
                        alpha = 0.3 + 0.7 * (j / (len(wake_x) - 1))
                        ax.plot([wake_x[j], wake_x[j+1]], [wake_y[j], wake_y[j+1]], 
                                'k-', linewidth=1.5, alpha=alpha)
                
                # Plot vessel (larger at the current position)
                ax.plot(vessel_x, vessel_y, 'ro', markersize=8)
                
                # Add time indicator
                if is_drift:
                    status = "DRIFTING"
                    power = 0
                else:
                    status = "POWERED"
                    power = power_profile[i] if i < len(power_profile) else 0
                
                # Show elapsed time and current status
                elapsed = t_idx - times[0]
                ax.set_title(f'Time: {t_idx:.1f} (Elapsed: {elapsed:.1f}), Status: {status}, Power: {power}%')
            
        # Add legend
        if frame_idx == 0:
            ax.legend()
        
        frames.append([plt.gcf()])
    
    # Create animation
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
    
    # Save if output file provided
    if output_file:
        print(f"Saving path animation to {output_file}...")
        ani.save(output_file, writer='ffmpeg')
    
    plt.tight_layout()
    plt.show()


def plot_drift_analysis(grid, time_idx=3):
    """
    Create a visualization focused on drift opportunities.
    
    This plots the current flow field and simulates drift paths from various points
    to show how the vessel would drift with no power at different positions.
    """
    from src.algorithms.spatio_temporal_astar import spatio_temporal_astar
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot environment at specified time
    plot_environment_state(ax, grid, time_idx, 'current')
    
    # Generate a set of points to analyze drift from
    stride = 10  # Space between points
    drift_points = []
    
    for x in range(stride, grid.grid_size[0]-stride, stride):
        for y in range(stride, grid.grid_size[1]-stride, stride):
            if not grid.is_obstacle(x, y):
                drift_points.append((x, y))
    
    # Simulate drifts from each point
    drift_duration = 3.0  # hours in time index units
    
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
            
            # Calculate drift distance and speed
            start_x, start_y = drift_xs[0], drift_ys[0]
            end_x, end_y = drift_xs[-1], drift_ys[-1]
            
            drift_dist = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            drift_speed = drift_dist / drift_duration
            
            # Skip very short drift paths
            if drift_dist < 2:
                continue
            
            # Color by drift speed
            color = plt.cm.plasma(drift_speed / 10)  # Normalize to reasonable range
            
            # Plot the drift path
            ax.plot(drift_xs, drift_ys, '-', color=color, linewidth=1.5, alpha=0.7)
            
            # Mark start and end
            ax.plot(start_x, start_y, 'o', color=color, markersize=4)
            ax.plot(end_x, end_y, 's', color=color, markersize=4)
    
    # Add colorbar for drift speed
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(0, 10))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Drift Speed (cells/hour)')
    
    # Set title
    phase_name = {0: "Flood Start", 6: "High Tide", 12: "Ebb Start", 18: "Low Tide"}
    ax.set_title(f"Drift Analysis at {phase_name.get(time_idx, f'Time {time_idx}')} (t={time_idx})")
    
    plt.tight_layout()
    plt.show()


def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Environmental pattern visualization')
    parser.add_argument('--demo', type=str, default='static',
                        choices=['static', 'animation', 'comparison', 'drift', 'path'],
                        help='Demo type to run')
    args = parser.parse_args()
    
    # Create the tidal channel environment
    print("Creating tidal channel environment...")
    grid_size = (60, 30)  # Reduced size for faster processing
    time_steps = 12  # 12 hours for faster processing
    field = create_tidal_channel(
        grid_size=grid_size, 
        time_steps=time_steps,
        tidal_period=12.42  # Realistic M2 tidal period in hours
    )
    
    # Create navigation grid with obstacles
    print("Creating navigation grid with obstacles...")
    grid = create_navigation_grid(field, grid_size)
    
    # Run the requested demo
    if args.demo == 'static' or args.demo == 'all':
        # Plot a few static time steps
        print("Generating static visualizations...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot at different phases of tide
        plot_environment_state(axes[0], grid, 0, 'combined')       # Start
        plot_environment_state(axes[1], grid, time_steps//4, 'combined')  # Quarter way
        plot_environment_state(axes[2], grid, time_steps//2, 'combined')  # Half way (opposing tide)
        
        plt.tight_layout()
        plt.show()
    
    if args.demo == 'animation' or args.demo == 'all':
        # Create animation of the tidal cycle
        print("Creating animation...")
        create_environment_animation(grid, "tidal_channel_animation.mp4", "combined")
        
        # Show the wind pattern separately
        print("Creating wind pattern animation...")
        create_environment_animation(grid, "wind_pattern_animation.mp4", "wind")
    
    if args.demo == 'comparison' or args.demo == 'all':
        # Compare paths at different tide states
        print("\nComparing paths at different tide states...")
        paths = plot_path_comparison(grid, "Effect of Tidal State on Path Planning")
    else:
        paths = None
    
    if args.demo == 'drift' or args.demo == 'all':
        # Plot drift analysis for a specific tide state
        print("\nAnalyzing drift opportunities...")
        plot_drift_analysis(grid, time_idx=0)  # At flood tide
        plot_drift_analysis(grid, time_idx=12)  # At ebb tide
    
    if args.demo == 'path' or args.demo == 'all':
        # If we need to calculate paths first
        if paths is None:
            print("\nCalculating paths for animation...")
            from src.algorithms.spatio_temporal_astar import spatio_temporal_astar
            energy_result = spatio_temporal_astar(
                grid, (10, 15), (90, 35),
                start_time_idx=0.0,
                usv_cruise_speed=1.0,
                usv_max_speed=1.5,
                optimization_mode='lowest_energy'
            )
        else:
            # Choose one of the energy-optimal paths from the comparison
            best_energy_index = 0
            for i, (time_result, energy_result) in enumerate(paths):
                if energy_result['feasible']:
                    best_energy_index = i
                    break
            
            if best_energy_index < len(paths):
                _, energy_result = paths[best_energy_index]
            else:
                print("No feasible energy-optimal paths found!")
                return
        
        # Create animation of a vessel following a path
        print("\nCreating path animation...")
        create_path_animation(grid, energy_result, output_file="path_animation.mp4")


if __name__ == "__main__":
    main()