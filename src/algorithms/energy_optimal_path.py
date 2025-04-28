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
    min_progress: float = -0.05,  # Allow slight negative progress for better drifting
    drift_range: int = 3         # Extended range for drifting connections
) -> nx.DiGraph:
    """
    Build a graph with multiple power settings per edge for energy optimization.
    Includes extended drift connections to better leverage favorable currents.
    
    Args:
        grid: Navigation grid with currents
        max_speed: Maximum speed of the USV in m/s
        power_levels: Number of power settings to consider 
        min_progress: Minimum progress required for powered movement
        drift_range: Extended range to check for drift connections
        
    Returns:
        Graph with energy-weighted edges
    """
    # Make sure networkx is imported in this scope
    import networkx as nx
    
    G = nx.DiGraph()
    
    # Get grid dimensions
    width, height = grid.grid_size
    
    # Define the 8 possible movement directions for regular movement
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
    
    # PART 1: Add regular movement edges with power options
    for y in range(height):
        for x in range(width):
            if grid.is_obstacle(x, y):
                continue
            
            # Convert cell coordinates to world coordinates for current lookup
            sx, sy = grid.cell_to_coords(x, y)
            
            # Get current vector at this position
            if grid.current_field:
                current_u, current_v = grid.current_field.get_vector_at_position(sx, sy)
                current_vector = (current_u, current_v)
                current_magnitude = np.sqrt(current_u**2 + current_v**2)
            else:
                current_vector = (0, 0)
                current_magnitude = 0
                
            # From this cell, check all 8 directions for regular movement
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Check if neighbor is within grid bounds and not an obstacle
                if (0 <= nx < width and 
                    0 <= ny < height and 
                    not grid.is_obstacle(nx, ny)):
                    
                    # Calculate movement vector and normalize
                    movement_vector = (dx, dy)
                    movement_norm = np.sqrt(dx**2 + dy**2)
                    normalized_movement = (dx/movement_norm, dy/movement_norm)
                    
                    # Get distance between cells in world coordinates
                    _, _, world_distance = grid.calculate_travel_cost(x, y, nx, ny, return_distance=True)
                    
                    # Try different power settings for this edge
                    best_energy = float('inf')
                    best_power = None
                    best_time = None
                    best_progress = 0
                    
                    for power in power_settings:
                        # For zero power, be more permissive about drift direction
                        if power == 0:
                            # For drift, only care if the current would carry us to the target
                            # Calculate where the current would take us
                            drift_time_estimate = world_distance / (current_magnitude + 0.01)  # Avoid div by zero
                            drift_x = sx + current_u * drift_time_estimate
                            drift_y = sy + current_v * drift_time_estimate
                            
                            # Convert endpoint location to grid coordinates
                            endpoint_x, endpoint_y = grid.coords_to_cell(drift_x, drift_y)
                            
                            # If drift would take us close to the target, allow it
                            if abs(endpoint_x - nx) <= 1 and abs(endpoint_y - ny) <= 1:
                                # Calculate actual energy and time
                                energy = 0  # No energy for drifting
                                ground_speed = current_magnitude
                                travel_time = world_distance / ground_speed if ground_speed > 0.05 else float('inf')
                                
                                # This is the most efficient option by definition (0 energy)
                                best_energy = energy
                                best_power = power
                                best_time = travel_time
                                # Calculate actual progress for info
                                progress_dot = (current_u * normalized_movement[0] + 
                                              current_v * normalized_movement[1])
                                best_progress = progress_dot
                                
                                # No need to check other power settings if drift works
                                break
                        else:
                            # Calculate energy consumption and resulting speed for powered movement
                            energy, ground_speed, ground_velocity = calculate_energy_consumption(
                                power, current_vector, normalized_movement, max_speed
                            )
                            
                            # Calculate dot product to check if we're making progress
                            progress_vector = (ground_velocity[0], ground_velocity[1])
                            progress_dot = (progress_vector[0] * normalized_movement[0] + 
                                           progress_vector[1] * normalized_movement[1])
                            
                            # For powered movement, ensure we're making forward progress
                            if progress_dot > 0:
                                # Calculate time to travel
                                travel_time = world_distance / ground_speed if ground_speed > 0 else float('inf')
                                
                                # Use energy as the cost
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
    
    # PART 2: Add extended drift edges for better current utilization
    if grid.current_field:
        # For each cell, check if we can drift to more distant cells
        for y in range(height):
            for x in range(width):
                if grid.is_obstacle(x, y):
                    continue
                
                # Get current vector at this position
                sx, sy = grid.cell_to_coords(x, y)
                current_u, current_v = grid.current_field.get_vector_at_position(sx, sy)
                current_magnitude = np.sqrt(current_u**2 + current_v**2)
                
                # Skip if current is too weak for useful drift
                if current_magnitude < 0.1:
                    continue
                
                # Normalize current direction
                if current_magnitude > 0:
                    current_dir_x = current_u / current_magnitude
                    current_dir_y = current_v / current_magnitude
                else:
                    continue
                
                # Check cells in the direction of the current
                for dist in range(2, drift_range + 1):
                    # Estimate where the current would take us
                    drift_x = int(x + round(dist * current_dir_x))
                    drift_y = int(y + round(dist * current_dir_y))
                    
                    # Check if this cell is valid and not an obstacle
                    if (0 <= drift_x < width and 
                        0 <= drift_y < height and 
                        not grid.is_obstacle(drift_x, drift_y)):
                        
                        # Calculate world coordinates and distance
                        end_sx, end_sy = grid.cell_to_coords(drift_x, drift_y)
                        world_distance = np.sqrt((end_sx - sx)**2 + (end_sy - sy)**2)
                        
                        # Estimate drift time
                        drift_time = world_distance / current_magnitude
                        
                        # For long distances, verify drift path doesn't hit obstacles
                        if dist > 2:
                            # Check intermediate points along the drift path
                            obstacle_hit = False
                            for step in range(1, dist):
                                # Check fraction of the way
                                frac = step / dist
                                check_x = int(x + round(step * current_dir_x))
                                check_y = int(y + round(step * current_dir_y))
                                if (0 <= check_x < width and 
                                    0 <= check_y < height and 
                                    grid.is_obstacle(check_x, check_y)):
                                    obstacle_hit = True
                                    break
                            
                            if obstacle_hit:
                                continue
                        
                        # Add a zero-energy drift edge
                        G.add_edge(
                            (x, y), (drift_x, drift_y),
                            weight=0,  # Zero energy for drifting
                            power=0,   # Zero power
                            time=drift_time,
                            progress=current_magnitude,
                            drift=True  # Mark as a drift edge
                        )
    
    return G
    
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