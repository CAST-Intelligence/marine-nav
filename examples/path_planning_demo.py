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


def create_test_current_field():
    """Create a test current field with various flow patterns."""
    grid_size = (100, 100)
    vf = VectorField(grid_size, x_range=(0, 10), y_range=(0, 10))
    
    # Create some interesting current patterns
    def u_func(x, y):
        # Circular vortex at (3, 7)
        dx1 = x - 3
        dy1 = y - 7
        dist1 = np.sqrt(dx1**2 + dy1**2)
        vortex = 0
        if dist1 > 0.1:
            vortex = -0.3 * dy1 / dist1 * np.exp(-0.1 * dist1)
        
        # Channel flow from left to right
        channel = 0.2 * np.exp(-(y - 5)**2 / 2)
        
        return vortex + channel
    
    def v_func(x, y):
        # Circular vortex at (3, 7)
        dx1 = x - 3
        dy1 = y - 7
        dist1 = np.sqrt(dx1**2 + dy1**2)
        vortex = 0
        if dist1 > 0.1:
            vortex = 0.3 * dx1 / dist1 * np.exp(-0.1 * dist1)
        
        # Add a vertical component in one area
        vertical = 0.1 * np.exp(-(x - 7)**2 / 1) * np.exp(-(y - 4)**2 / 1)
        
        return vortex + vertical
    
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
    usv_speed = 0.5  # m/s
    
    # Find path with and without considering currents
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
    
    # Create search patterns
    # Expanding square
    square_center = (50, 70)
    square_pattern = generate_expanding_square_pattern(
        square_center, max_distance=15, step_size=3
    )
    
    # Sector search
    sector_center = (70, 40)
    sector_pattern = generate_sector_search_pattern(
        sector_center, max_distance=12, num_legs=7, orientation=0.2
    )
    
    # Parallel search
    parallel_start = (40, 40)
    parallel_pattern = generate_parallel_search_pattern(
        parallel_start, width=30, height=20, spacing=4, orientation=0.3
    )
    
    # Plot the results
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot A* paths
    plot_navigation_grid(
        nav_grid,
        ax=axes[0, 0],
        show_obstacles=True,
        show_currents=True,
        path=path_with_currents,
        start_point=start,
        end_point=goal,
        title="A* Path Planning With Current Awareness"
    )
    
    plot_navigation_grid(
        nav_grid,
        ax=axes[0, 1],
        show_obstacles=True,
        show_currents=True,
        path=path_without_currents,
        start_point=start,
        end_point=goal,
        title="A* Path Planning Without Current Awareness"
    )
    
    # Plot search patterns
    plot_navigation_grid(
        nav_grid,
        ax=axes[1, 0],
        show_obstacles=True,
        show_currents=True,
        path=square_pattern + sector_pattern,
        start_point=square_center,
        title="Search Patterns: Expanding Square and Sector Search"
    )
    
    plot_navigation_grid(
        nav_grid,
        ax=axes[1, 1],
        show_obstacles=True,
        show_currents=True,
        path=parallel_pattern,
        start_point=parallel_start,
        title="Search Pattern: Parallel Search"
    )
    
    plt.tight_layout()
    plt.savefig("path_planning_demo.png", dpi=150)
    plt.show()
    
    print("Path planning with currents demonstration complete.")
    print(f"Path length with currents: {len(path_with_currents)} cells")
    print(f"Path length without currents: {len(path_without_currents)} cells")


if __name__ == "__main__":
    main()