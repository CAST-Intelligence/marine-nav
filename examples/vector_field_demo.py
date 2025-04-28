"""
Example demonstrating current vector field in marine navigation.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.vector_field import VectorField
from src.core.grid import NavigationGrid
from src.visualization.grid_viz import plot_vector_field, plot_navigation_grid


def create_circular_current(
    x_center: float, 
    y_center: float, 
    strength: float = 1.0, 
    falloff: float = 0.1
):
    """
    Create a circular current centered at (x_center, y_center).
    
    Args:
        x_center: x-coordinate of center
        y_center: y-coordinate of center
        strength: Current strength factor
        falloff: Strength reduction with distance
    
    Returns:
        Functions for u and v components
    """
    def u_func(x, y):
        dx = x - x_center
        dy = y - y_center
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 1e-6:
            return 0
        factor = strength * np.exp(-falloff * dist)
        return -dy / dist * factor
    
    def v_func(x, y):
        dx = x - x_center
        dy = y - y_center
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 1e-6:
            return 0
        factor = strength * np.exp(-falloff * dist)
        return dx / dist * factor
    
    return u_func, v_func


def create_channel_current(
    direction: float = 0.0,  # In radians, 0 = east, pi/2 = north
    strength: float = 1.0,
    width: float = 0.3,  # Width of the channel
    center: float = 0.5,  # Center of the channel
):
    """
    Create a channel current flowing along a specified direction.
    
    Args:
        direction: Direction of flow in radians
        strength: Current strength
        width: Width of the channel
        center: Center position of the channel orthogonal to flow
    
    Returns:
        Functions for u and v components
    """
    # Unit vector in flow direction
    ux = np.cos(direction)
    uy = np.sin(direction)
    
    # Orthogonal unit vector (for channel width)
    ox = -uy
    oy = ux
    
    def u_func(x, y):
        # Project point onto orthogonal axis
        proj = x * ox + y * oy
        # Distance from center line
        dist = abs(proj - center)
        # Gaussian profile across the channel
        profile = np.exp(-(dist**2) / (2 * width**2))
        return ux * strength * profile
    
    def v_func(x, y):
        # Project point onto orthogonal axis
        proj = x * ox + y * oy
        # Distance from center line
        dist = abs(proj - center)
        # Gaussian profile across the channel
        profile = np.exp(-(dist**2) / (2 * width**2))
        return uy * strength * profile
    
    return u_func, v_func


def main():
    # Create a vector field for currents
    grid_size = (100, 100)
    vf = VectorField(grid_size, x_range=(0, 10), y_range=(0, 10))
    
    # Create a combination of current patterns
    # 1. A vortex current
    u_vortex, v_vortex = create_circular_current(3, 7, strength=0.5, falloff=0.1)
    
    # 2. A channel current flowing west to east
    u_channel, v_channel = create_channel_current(direction=0.0, strength=0.3, center=5, width=1.0)
    
    # Combine the currents
    def u_combined(x, y):
        return u_vortex(x, y) + u_channel(x, y)
    
    def v_combined(x, y):
        return v_vortex(x, y) + v_channel(x, y)
    
    # Generate the vector field
    vf.generate_from_function(u_combined, v_combined)
    
    # Create a navigation grid
    nav_grid = NavigationGrid(grid_size, cell_size=0.1, x_origin=0, y_origin=0)
    
    # Set the current field in the navigation grid
    nav_grid.set_current_field(vf)
    
    # Add some obstacles
    # Island 1
    nav_grid.add_obstacle_region(20, 20, 35, 35)
    
    # Island 2 - Elongated
    nav_grid.add_obstacle_region(60, 50, 80, 60)
    
    # Coastal area
    for x in range(grid_size[0]):
        for y in range(10):
            nav_grid.add_obstacle(x, y)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot the vector field
    plot_vector_field(vf, ax=ax1, title="Marine Current Vector Field")
    
    # Plot the navigation grid with currents
    plot_navigation_grid(
        nav_grid, 
        ax=ax2, 
        show_obstacles=True,
        show_currents=True,
        title="Navigation Grid with Currents and Obstacles"
    )
    
    plt.tight_layout()
    plt.savefig("current_visualization.png", dpi=150)
    plt.show()
    
    print("Vector field and navigation grid created and visualized.")
    print("Next steps would include path planning algorithms that account for currents.")


if __name__ == "__main__":
    main()