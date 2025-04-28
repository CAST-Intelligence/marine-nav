"""
Graph-based path planning using network topology for USV navigation.
"""
from typing import List, Tuple, Dict, Optional
import networkx as nx
import numpy as np
from ..core.grid import NavigationGrid


def build_graph_from_grid(
    grid: NavigationGrid, 
    usv_speed: float = 1.0
) -> nx.DiGraph:
    """
    Build a directed graph from a navigation grid, with edges in 8 directions
    and costs based on travel time accounting for currents.
    
    Args:
        grid: The navigation grid with current field
        usv_speed: The speed of the USV in m/s
        
    Returns:
        A NetworkX DiGraph representing the navigable grid
    """
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
    
    # Add edges with travel time costs
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
                    
                    # Calculate travel cost accounting for currents
                    cost = grid.calculate_travel_cost(x, y, nx, ny, usv_speed)
                    
                    # Add directed edge with cost
                    G.add_edge((x, y), (nx, ny), weight=cost)
    
    return G


def find_shortest_time_path(
    grid: NavigationGrid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    usv_speed: float = 1.0
) -> List[Tuple[int, int]]:
    """
    Find the shortest time path using Dijkstra's algorithm on a graph network
    derived from the grid, with edge weights representing travel time.
    
    Args:
        grid: The navigation grid with current field
        start: Starting point as (x, y) cell coordinates
        goal: Goal point as (x, y) cell coordinates
        usv_speed: The speed of the USV in m/s
        
    Returns:
        List of (x, y) cell coordinates forming the path
    """
    # Build graph from grid
    G = build_graph_from_grid(grid, usv_speed)
    
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
    
    # Use Dijkstra's algorithm to find shortest path based on time costs
    try:
        path = nx.shortest_path(G, source=start, target=goal, weight='weight')
        return path
    except nx.NetworkXNoPath:
        # No path found
        return []