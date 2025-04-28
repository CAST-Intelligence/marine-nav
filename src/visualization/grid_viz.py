"""
Visualization tools for grid and vector field.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, List, Optional
from ..core.grid import NavigationGrid
from ..core.vector_field import VectorField


def plot_vector_field(
    vector_field: VectorField,
    ax=None,
    density: int = 20,
    color: str = 'blue',
    scale: float = 1.0,
    show_magnitude: bool = True,
    title: str = "Current Vector Field"
):
    """
    Plot a vector field on the given axes.
    
    Args:
        vector_field: The VectorField to plot
        ax: Matplotlib axes to plot on (creates new figure if None)
        density: Number of arrows to show in each dimension
        color: Arrow color
        scale: Scaling factor for arrow sizes
        show_magnitude: Whether to show magnitude as color intensity
        title: Plot title
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a sparser grid for visualization
    nx, ny = vector_field.grid_size
    x_range = vector_field.x_range
    y_range = vector_field.y_range
    
    step_x = max(1, nx // density)
    step_y = max(1, ny // density)
    
    X_vis = vector_field.X[::step_y, ::step_x]
    Y_vis = vector_field.Y[::step_y, ::step_x]
    U_vis = vector_field.U[::step_y, ::step_x]
    V_vis = vector_field.V[::step_y, ::step_x]
    
    # Calculate magnitudes for coloring if requested
    if show_magnitude:
        magnitude = np.sqrt(U_vis**2 + V_vis**2)
        color_array = magnitude / np.max(magnitude) if np.max(magnitude) > 0 else magnitude
        quiver = ax.quiver(X_vis, Y_vis, U_vis, V_vis, color_array, 
                        cmap='Blues', scale=1.0/scale, scale_units='xy', 
                        pivot='mid', width=0.003, headwidth=4, headlength=4)
        plt.colorbar(quiver, ax=ax, label='Current Speed')
    else:
        ax.quiver(X_vis, Y_vis, U_vis, V_vis, color=color, 
                scale=1.0/scale, scale_units='xy', 
                pivot='mid', width=0.003, headwidth=4, headlength=4)
    
    # Plot critical points if available
    try:
        critical_points = vector_field.find_critical_points()
        if critical_points:
            cp_colors = {
                'source': 'red',
                'sink': 'green',
                'saddle': 'yellow',
                'center': 'purple',
                'spiral-source': 'orange',
                'spiral-sink': 'cyan'
            }
            
            for x, y, cp_type in critical_points:
                ax.plot(x, y, 'o', color=cp_colors.get(cp_type, 'white'), 
                      markersize=8, label=cp_type)
            
            # Create legend with unique entries
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), 
                    title="Critical Points", loc='upper right')
    except:
        pass
    
    # Set grid properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    return ax


def plot_navigation_grid(
    nav_grid: NavigationGrid, 
    ax=None, 
    show_obstacles: bool = True,
    show_costs: bool = False,
    show_currents: bool = True,
    current_density: int = 15,
    path: Optional[List[Tuple[int, int]]] = None,
    start_point: Optional[Tuple[int, int]] = None,
    end_point: Optional[Tuple[int, int]] = None,
    title: str = "Navigation Grid"
):
    """
    Plot a navigation grid with optional path and currents.
    
    Args:
        nav_grid: The NavigationGrid to plot
        ax: Matplotlib axes to plot on (creates new figure if None)
        show_obstacles: Whether to show obstacles
        show_costs: Whether to show cost values as a heatmap
        show_currents: Whether to show the current field
        current_density: Density of current vectors to display
        path: Optional path to display as (x,y) cell coordinates
        start_point: Optional start point as (x,y) cell coordinates
        end_point: Optional end point as (x,y) cell coordinates
        title: Plot title
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
        
    # Plot base grid
    x_size, y_size = nav_grid.grid_size
    extent = [
        nav_grid.x_origin, 
        nav_grid.x_origin + x_size * nav_grid.cell_size,
        nav_grid.y_origin, 
        nav_grid.y_origin + y_size * nav_grid.cell_size
    ]
    
    # Create a grid background
    if show_costs and not show_obstacles:
        # Show costs as heatmap
        im = ax.imshow(nav_grid.cost_grid, origin='lower', extent=extent, 
                     alpha=0.7, cmap='YlOrRd')
        plt.colorbar(im, ax=ax, label='Cost')
    else:
        # Show a light grid
        ax.pcolormesh(
            nav_grid.X, nav_grid.Y, 
            np.zeros(nav_grid.grid_size), 
            edgecolors='lightgray', linewidth=0.5,
            facecolor='white', alpha=0.3
        )
    
    # Show obstacles
    if show_obstacles:
        obstacle_mask = nav_grid.obstacle_grid
        obstacle_x = []
        obstacle_y = []
        
        for x in range(x_size):
            for y in range(y_size):
                if obstacle_mask[y, x]:
                    # Convert to world coordinates
                    world_x, world_y = nav_grid.cell_to_coords(x, y)
                    obstacle_x.append(world_x)
                    obstacle_y.append(world_y)
        
        if obstacle_x:
            # Plot obstacles as scatter points for efficiency
            ax.scatter(obstacle_x, obstacle_y, color='black', marker='s', 
                      s=nav_grid.cell_size*100, label='Obstacles')
    
    # Show current vector field
    if show_currents and nav_grid.current_field is not None:
        plot_vector_field(
            nav_grid.current_field, 
            ax=ax, 
            density=current_density,
            scale=2.0,
            show_magnitude=True,
            title=None
        )
    
    # Plot path if provided
    if path:
        path_x = []
        path_y = []
        
        for x, y in path:
            world_x, world_y = nav_grid.cell_to_coords(x, y)
            path_x.append(world_x)
            path_y.append(world_y)
        
        ax.plot(path_x, path_y, 'o-', color='green', linewidth=2, markersize=4, label='Path')
    
    # Plot start and end points
    if start_point:
        start_x, start_y = nav_grid.cell_to_coords(*start_point)
        ax.plot(start_x, start_y, 'o', color='blue', markersize=10, label='Start')
    
    if end_point:
        end_x, end_y = nav_grid.cell_to_coords(*end_point)
        ax.plot(end_x, end_y, 'o', color='red', markersize=10, label='End')
    
    # Set grid properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend if we have elements to show
    if path or start_point or end_point or show_obstacles:
        ax.legend(loc='upper right')
    
    return ax