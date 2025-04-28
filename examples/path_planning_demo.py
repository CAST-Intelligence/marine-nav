"""
Example demonstrating path planning with currents.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.vector_field import VectorField
from src.core.grid import NavigationGrid
from src.visualization.grid_viz import plot_navigation_grid
from src.algorithms.path_planning import (
    a_star_current_aware,
    generate_expanding_square_pattern,
    generate_sector_search_pattern,
    generate_parallel_search_pattern
)
from src.algorithms.network_path_finding import find_shortest_time_path
from src.algorithms.energy_optimal_path import find_energy_optimal_path


def create_test_current_field():
    """Create a test current field with various flow patterns."""
    grid_size = (100, 100)
    vf = VectorField(grid_size, x_range=(0, 10), y_range=(0, 10))
    
    # Create some interesting current patterns with increased strength
    def u_func(x, y):
        # Circular vortex at (3, 7) - stronger rotation
        dx1 = x - 3
        dy1 = y - 7
        dist1 = np.sqrt(dx1**2 + dy1**2)
        vortex = 0
        if dist1 > 0.1:
            vortex = -0.8 * dy1 / dist1 * np.exp(-0.1 * dist1)
        
        # Strong channel flow from left to right
        channel = 0.6 * np.exp(-(y - 5)**2 / 2)
        
        # Add a small counter-current in the upper area
        counter = -0.3 * np.exp(-(y - 8)**2 / 1)
        
        return vortex + channel + counter
    
    def v_func(x, y):
        # Circular vortex at (3, 7) - stronger rotation
        dx1 = x - 3
        dy1 = y - 7
        dist1 = np.sqrt(dx1**2 + dy1**2)
        vortex = 0
        if dist1 > 0.1:
            vortex = 0.8 * dx1 / dist1 * np.exp(-0.1 * dist1)
        
        # Add a stronger vertical component in one area
        vertical = 0.4 * np.exp(-(x - 7)**2 / 1) * np.exp(-(y - 4)**2 / 1)
        
        # Add a diagonal flow component in the bottom right
        diagonal = 0.3 * np.exp(-(x - 8)**2 / 4) * np.exp(-(y - 2)**2 / 4)
        
        return vortex + vertical + diagonal
    
    vf.generate_from_function(u_func, v_func)
    
    return vf


def main():
    # Create a navigation grid with current field
    grid_size = (100, 100)
    nav_grid = NavigationGrid(grid_size, cell_size=0.1, x_origin=0, y_origin=0)
    
    # Set the current field
    current_field = create_test_current_field()
    nav_grid.set_current_field(current_field)
    
    # Add some obstacles
    # Island 1
    nav_grid.add_obstacle_region(20, 20, 35, 35)
    
    # Island 2 - Elongated
    nav_grid.add_obstacle_region(60, 50, 80, 60)
    
    # Coastal area
    for x in range(grid_size[0]):
        for y in range(10):
            nav_grid.add_obstacle(x, y)
    
    # Define start and goal points
    start = (15, 80)  # Top left
    goal = (85, 30)   # Bottom right
    
    # Plan a path using A* that accounts for currents
    usv_speed = 0.7  # m/s - set speed comparable to currents to make effect more obvious
    
    # Find path with and without considering currents using A*
    path_with_currents = a_star_current_aware(
        nav_grid, start, goal, usv_speed=usv_speed
    )
    
    # Temporarily remove current field to find path without considering currents
    temp_field = nav_grid.current_field
    nav_grid.current_field = None
    path_without_currents = a_star_current_aware(
        nav_grid, start, goal, usv_speed=usv_speed
    )
    nav_grid.current_field = temp_field
    
    # Use graph-based Dijkstra's algorithm for comparison with A*
    dijkstra_path = find_shortest_time_path(
        nav_grid, start, goal, usv_speed=usv_speed
    )
    
    # Use energy-optimized path planning
    energy_path, power_settings = find_energy_optimal_path(
        nav_grid, start, goal, max_speed=usv_speed, power_levels=6
    )
    
    # Print power settings for energy-optimal path
    if energy_path and power_settings:
        print("\nEnergy-optimal path power settings:")
        # Group consecutive identical power settings
        current_power = power_settings[0]
        count = 1
        for power in power_settings[1:]:
            if power == current_power:
                count += 1
            else:
                print(f"  - {count} segments at {current_power}% power")
                current_power = power
                count = 1
        print(f"  - {count} segments at {current_power}% power")
    
    # Create IAMSAR-compliant search patterns with obstacle awareness
    # Expanding square (spiral pattern)
    square_center = (50, 70)
    square_pattern = generate_expanding_square_pattern(
        square_center, max_distance=30, step_size=5, grid=nav_grid
    )
    
    # Sector search with triangular sweeps
    sector_center = (70, 40)
    sector_pattern = generate_sector_search_pattern(
        sector_center, max_distance=15, num_triangles=6, 
        initial_orientation=0.2, triangle_angle=np.pi/3, grid=nav_grid
    )
    
    # Parallel search with geometrically accurate obstacle avoidance
    parallel_start = (20, 40)  # Moved further left to better demonstrate coverage around obstacles
    parallel_pattern = generate_parallel_search_pattern(
        parallel_start, width=55, height=30, spacing=5, orientation=0.0, grid=nav_grid
    )
    
    # Plot the results
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot both A* and Dijkstra paths for comparison
    ax_comparison = plot_navigation_grid(
        nav_grid,
        ax=axes[0, 0],
        show_obstacles=True,
        show_currents=True,
        path=path_with_currents,
        start_point=start,
        end_point=goal,
        title="Path Optimization Comparison\n(A*: Green circles, Dijkstra: Purple triangles, Energy-optimal: Color-coded by power)"
    )
    
    # Add Dijkstra path in a different color with triangular markers
    if dijkstra_path:
        # Convert to world coordinates for plotting
        dijkstra_world_points = []
        for x, y in dijkstra_path:
            world_x, world_y = nav_grid.cell_to_coords(x, y)
            dijkstra_world_points.append((world_x, world_y))
        
        # Extract x and y coordinates
        d_x = [p[0] for p in dijkstra_world_points]
        d_y = [p[1] for p in dijkstra_world_points]
        
        # Plot Dijkstra path with triangular markers
        # Line only (no markers) for the path
        ax_comparison.plot(d_x, d_y, '-', color='purple', linewidth=2, label='Time-optimal Dijkstra')
        
        # Add triangular markers on top
        ax_comparison.scatter(d_x, d_y, marker='^', color='purple', s=80, 
                            edgecolor='black', linewidth=0.5)
    
    # Add Energy-optimized path with color-coded square markers based on power level
    if energy_path and power_settings:
        # Create a colormap for power settings (0-100%)
        import matplotlib.cm as cm
        power_cmap = cm.get_cmap('plasma')
        
        # Line only (no markers) for the path
        # Convert to world coordinates for plotting
        energy_world_points = []
        for x, y in energy_path:
            world_x, world_y = nav_grid.cell_to_coords(x, y)
            energy_world_points.append((world_x, world_y))
        
        # Extract x and y coordinates
        e_x = [p[0] for p in energy_world_points]
        e_y = [p[1] for p in energy_world_points]
        
        # Plot the path line
        ax_comparison.plot(e_x, e_y, '-', color='black', linewidth=1.5, alpha=0.5, label='Energy-optimal path')
        
        # Add square markers with colors based on power level
        for i in range(len(energy_path) - 1):
            if i < len(power_settings):
                # Get power setting for this segment
                power = power_settings[i]
                
                # Normalize power to 0-1 for colormap
                normalized_power = power / 100.0
                
                # Get color from colormap
                color = power_cmap(normalized_power)
                
                # Get world coordinates for this point
                x, y = energy_path[i]
                world_x, world_y = nav_grid.cell_to_coords(x, y)
                
                # Plot square marker with power-based color (smaller size)
                ax_comparison.scatter(world_x, world_y, marker='s', color=color, s=40, 
                                    edgecolor='black', linewidth=0.5)
        
        # Add a colorbar to show power levels
        sm = plt.cm.ScalarMappable(cmap=power_cmap, norm=plt.Normalize(0, 100))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_comparison, label='Power Level (%)', shrink=0.6)
        
        # Add a text label for the energy path
        ax_comparison.text(0.05, 0.05, 'Energy-optimal path (colored squares = power level)', 
                         transform=ax_comparison.transAxes, fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.7))
        
    # Update legend
    ax_comparison.legend(loc='lower right')
    
    plot_navigation_grid(
        nav_grid,
        ax=axes[0, 1],
        show_obstacles=True,
        show_currents=True,
        path=path_without_currents,
        start_point=start,
        end_point=goal,
        title="Shortest Distance Path Ignoring Currents\n(Likely higher energy consumption)"
    )
    
    # Plot search patterns
    plot_navigation_grid(
        nav_grid,
        ax=axes[1, 0],
        show_obstacles=True,
        show_currents=True,
        path=square_pattern + sector_pattern,
        start_point=square_center,
        title="IAMSAR Search Patterns: Expanding Square and Sector Search\n(Obstacle-aware with pattern integrity preservation)"
    )
    
    plot_navigation_grid(
        nav_grid,
        ax=axes[1, 1],
        show_obstacles=True,
        show_currents=True,
        path=parallel_pattern,
        start_point=parallel_start,
        title="IAMSAR Search Pattern: Parallel Search\n(Using geometric operations for complete area coverage)"
    )
    
    plt.tight_layout()
    plt.savefig("path_planning_demo.png", dpi=150)
    plt.show()
    
    # Calculate metrics for all paths
    def calculate_path_metrics(path, grid, usv_speed):
        """Calculate total distance and time for a path."""
        total_distance = 0
        total_time = 0
        
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            # Convert to world coordinates
            wx1, wy1 = grid.cell_to_coords(x1, y1)
            wx2, wy2 = grid.cell_to_coords(x2, y2)
            
            # Calculate Euclidean distance
            distance = np.sqrt((wx2 - wx1)**2 + (wy2 - wy1)**2)
            
            # Calculate travel time using the grid's cost function
            time = grid.calculate_travel_cost(x1, y1, x2, y2, usv_speed)
            
            total_distance += distance
            total_time += time
            
        return total_distance, total_time
    
    # Define function to calculate energy consumption
    def calculate_energy_consumption(path, grid, usv_speed):
        """Calculate total energy consumption for a path using fixed speed."""
        total_energy = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            # Get world coordinates for current vector
            wx1, wy1 = grid.cell_to_coords(x1, y1)
            
            # Calculate movement vector
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                movement_vector = (dx/dist, dy/dist)
            else:
                movement_vector = (0, 0)
            
            # Get current vector at this position
            if grid.current_field:
                current_u, current_v = grid.current_field.get_vector_at_position(wx1, wy1)
                current_vector = (current_u, current_v)
            else:
                current_vector = (0, 0)
            
            # Calculate energy using cubic relationship with power
            # Assume 100% power for fixed speed methods
            segment_energy = 1.0 * ((usv_speed / max_speed) ** 3)
            total_energy += segment_energy
            
        return total_energy
    
    # Fixed max speed for comparison
    max_speed = 1.0
    
    # Calculate metrics for A* path
    a_star_distance, a_star_time = calculate_path_metrics(
        path_with_currents, nav_grid, usv_speed
    )
    a_star_energy = calculate_energy_consumption(path_with_currents, nav_grid, usv_speed)
    
    # Calculate metrics for Dijkstra path
    dijkstra_distance, dijkstra_time = calculate_path_metrics(
        dijkstra_path, nav_grid, usv_speed
    )
    dijkstra_energy = calculate_energy_consumption(dijkstra_path, nav_grid, usv_speed)
    
    # Calculate metrics for energy-optimal path
    if energy_path:
        energy_distance, energy_time = calculate_path_metrics(
            energy_path, nav_grid, usv_speed
        )
        
        # For energy path, use the actual power settings for energy calculation
        energy_optimal_energy = 0
        for i, power in enumerate(power_settings):
            if i < len(energy_path) - 1:
                # Energy consumption is proportional to power^3
                energy_optimal_energy += (power / 100.0) ** 3
    else:
        energy_distance = 0
        energy_time = 0
        energy_optimal_energy = 0
    
    # Calculate metrics for path without currents
    no_current_distance, _ = calculate_path_metrics(
        path_without_currents, nav_grid, usv_speed
    )
    
    print("Path planning with currents demonstration complete.")
    print(f"A* Path: {len(path_with_currents)} cells, Distance: {a_star_distance:.2f}m, Time: {a_star_time:.2f}s, Energy: {a_star_energy:.2f}")
    print(f"Dijkstra Path: {len(dijkstra_path)} cells, Distance: {dijkstra_distance:.2f}m, Time: {dijkstra_time:.2f}s, Energy: {dijkstra_energy:.2f}")
    print(f"Energy-optimal Path: {len(energy_path)} cells, Distance: {energy_distance:.2f}m, Time: {energy_time:.2f}s, Energy: {energy_optimal_energy:.2f}")
    print(f"Direct Path (ignoring currents): {len(path_without_currents)} cells, Distance: {no_current_distance:.2f}m")


if __name__ == "__main__":
    main()