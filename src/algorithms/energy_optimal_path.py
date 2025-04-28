"""
Energy-optimized path planning for USVs in current fields.

This module implements path planning that minimizes energy consumption
by varying power levels and leveraging favorable currents.
"""
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import networkx as nx
from ..core.grid import NavigationGrid


def calculate_energy_consumption(
    usv_power: float,
    current_vector: Tuple[float, float],
    movement_vector: Tuple[float, float],
    max_speed: float = 1.0,
    energy_factor: float = 1.0
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Calculate energy consumption based on motor power and currents.
    
    Args:
        usv_power: Power setting of the USV (0-100%)
        current_vector: Vector of water current (u, v)
        movement_vector: Desired movement direction (normalized)
        max_speed: Maximum speed of the USV in m/s
        energy_factor: Scaling factor for energy consumption
        
    Returns:
        Tuple of (energy_consumption, ground_speed, ground_velocity_vector)
    """
    # Convert power to speed through water
    usv_speed = max_speed * (usv_power / 100.0)
    
    # Calculate resulting velocity vector (boat + current)
    ground_velocity_x = movement_vector[0] * usv_speed + current_vector[0]
    ground_velocity_y = movement_vector[1] * usv_speed + current_vector[1]
    
    # Calculate ground speed
    ground_speed = np.sqrt(ground_velocity_x**2 + ground_velocity_y**2)
    
    # Energy consumption model (simplified)
    # Power curve - energy is proportional to power^3
    energy = (usv_power / 100.0)**3 * energy_factor
    
    return energy, ground_speed, (ground_velocity_x, ground_velocity_y)


def build_energy_optimized_graph(
    grid: NavigationGrid,
    max_speed: float = 1.0,
    power_levels: int = 5,
    min_progress: float = 0.05
) -> nx.DiGraph:
    """
    Build a graph with multiple power settings per edge for energy optimization.
    
    Args:
        grid: Navigation grid with currents
        max_speed: Maximum speed of the USV in m/s
        power_levels: Number of power settings to consider 
        min_progress: Minimum forward progress required (as fraction of max_speed)
        
    Returns:
        Graph with energy-weighted edges
    """
    # Make sure networkx is imported in this scope
    import networkx as nx
    
    G = nx.DiGraph()
    
    # Get grid dimensions
    width, height = grid.grid_size
    
    # Define the 8 possible movement directions
    directions = [
        (1, 0),   # Right
        (1, 1),   # Up-Right
        (0, 1),   # Up
        (-1, 1),  # Up-Left
        (-1, 0),  # Left
        (-1, -1), # Down-Left
        (0, -1),  # Down
        (1, -1)   # Down-Right
    ]
    
    # Add nodes for all navigable cells
    for y in range(height):
        for x in range(width):
            if not grid.is_obstacle(x, y):
                G.add_node((x, y))
    
    # Power settings to try (0% means drift with current)
    power_settings = [0] + list(range(20, 101, int(80/(power_levels-1)))) if power_levels > 1 else [100]
    
    # Add edges with energy costs for each power setting
    for y in range(height):
        for x in range(width):
            if grid.is_obstacle(x, y):
                continue
                
            # From this cell, check all 8 directions
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Check if neighbor is within grid bounds and not an obstacle
                if (0 <= nx < width and 
                    0 <= ny < height and 
                    not grid.is_obstacle(nx, ny)):
                    
                    # Convert cell coordinates to world coordinates for current lookup
                    sx, sy = grid.cell_to_coords(x, y)
                    
                    # Calculate movement vector and normalize
                    movement_vector = (dx, dy)
                    movement_norm = np.sqrt(dx**2 + dy**2)
                    normalized_movement = (dx/movement_norm, dy/movement_norm)
                    
                    # Get current vector at this position
                    if grid.current_field:
                        current_u, current_v = grid.current_field.get_vector_at_position(sx, sy)
                        current_vector = (current_u, current_v)
                    else:
                        current_vector = (0, 0)
                    
                    # Get distance between cells in world coordinates
                    _, _, world_distance = grid.calculate_travel_cost(x, y, nx, ny, return_distance=True)
                    
                    # Try different power settings
                    best_energy = float('inf')
                    best_power = None
                    best_time = None
                    best_progress = 0
                    
                    for power in power_settings:
                        # Calculate energy consumption and resulting speed
                        energy, ground_speed, ground_velocity = calculate_energy_consumption(
                            power, current_vector, normalized_movement, max_speed
                        )
                        
                        # Calculate dot product to check if we're making progress in desired direction
                        progress_vector = (ground_velocity[0], ground_velocity[1])
                        progress_dot = (progress_vector[0] * normalized_movement[0] + 
                                       progress_vector[1] * normalized_movement[1])
                        
                        # Only add edge if we make sufficient forward progress
                        if progress_dot > min_progress * max_speed:
                            # Calculate time to travel
                            travel_time = world_distance / ground_speed if ground_speed > 0 else float('inf')
                            
                            # Use weighted combination of energy and time
                            total_cost = energy
                            
                            # Check if this is the most efficient setting so far
                            if total_cost < best_energy:
                                best_energy = total_cost
                                best_power = power
                                best_time = travel_time
                                best_progress = progress_dot
                    
                    # Add the edge with the best power setting
                    if best_power is not None:
                        G.add_edge(
                            (x, y), (nx, ny),
                            weight=best_energy,  # Use energy as primary weight
                            power=best_power,
                            time=best_time,
                            progress=best_progress
                        )
    
    return G


def find_energy_optimal_path(
    grid: NavigationGrid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_speed: float = 1.0,
    power_levels: int = 5
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Find path that minimizes total energy consumption.
    
    Args:
        grid: Navigation grid with currents
        start: Starting point as (x, y) cell coordinates
        goal: Goal point as (x, y) cell coordinates
        max_speed: Maximum USV speed in m/s
        power_levels: Number of power settings to consider
        
    Returns:
        Tuple of (path, power_settings)
    """
    # Make sure networkx is imported in this scope
    import networkx as nx
    
    # Build multi-power graph
    G = build_energy_optimized_graph(grid, max_speed, power_levels)
    
    # Check if start and goal are in the graph
    if start not in G:
        # Find closest navigable node to start
        closest_dist = float('inf')
        closest_node = None
        for node in G.nodes():
            dist = (node[0] - start[0])**2 + (node[1] - start[1])**2
            if dist < closest_dist:
                closest_dist = dist
                closest_node = node
        start = closest_node
    
    if goal not in G:
        # Find closest navigable node to goal
        closest_dist = float('inf')
        closest_node = None
        for node in G.nodes():
            dist = (node[0] - goal[0])**2 + (node[1] - goal[1])**2
            if dist < closest_dist:
                closest_dist = dist
                closest_node = node
        goal = closest_node
    
    # Use Dijkstra to find minimum energy path
    try:
        path = nx.shortest_path(G, source=start, target=goal, weight='weight')
        
        # Extract power settings for each segment
        power_settings = []
        for i in range(len(path) - 1):
            power = G[path[i]][path[i+1]].get('power', 100)
            power_settings.append(power)
        
        return path, power_settings
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # No path found
        return [], []