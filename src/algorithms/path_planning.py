"""
Path planning algorithms for USV navigation with currents.
"""
from typing import List, Tuple, Dict, Set, Optional, Any
import heapq
import numpy as np
from ..core.grid import NavigationGrid


def a_star_current_aware(
    grid: NavigationGrid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    usv_speed: float = 1.0,
    heuristic_weight: float = 1.0
) -> List[Tuple[int, int]]:
    """
    A* algorithm for path planning that accounts for currents.
    
    Args:
        grid: The navigation grid with current information
        start: Starting point as (x, y) cell coordinates
        goal: Goal point as (x, y) cell coordinates
        usv_speed: The speed of the USV in m/s
        heuristic_weight: Weight for the heuristic (1.0 = standard A*)
        
    Returns:
        List of (x, y) cell coordinates forming the path
    """
    # Initialize open and closed sets
    open_set = []
    closed_set = set()
    
    # The g_score is the cost from start to current node
    g_score = {start: 0}
    
    # The f_score is g_score + heuristic (estimated cost to goal)
    f_score = {start: heuristic(grid, start, goal)}
    
    # The open set is implemented as a priority queue
    heapq.heappush(open_set, (f_score[start], start))
    
    # Dictionary to reconstruct path
    came_from = {}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            return reconstruct_path(came_from, current)
        
        closed_set.add(current)
        
        for neighbor in grid.get_neighbors(*current):
            if neighbor in closed_set:
                continue
                
            # Calculate the cost to neighbor with currents
            tentative_g_score = g_score[current] + grid.calculate_travel_cost(
                current[0], current[1], neighbor[0], neighbor[1], usv_speed
            )
            
            # If this is a new node or we found a better path
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic_weight * heuristic(grid, neighbor, goal)
                
                # Check if we need to add to open set
                if not any(neighbor == node for _, node in open_set):
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # If we get here, no path was found
    return []


def heuristic(
    grid: NavigationGrid,
    current: Tuple[int, int],
    goal: Tuple[int, int]
) -> float:
    """
    Calculate heuristic for A* (Euclidean distance).
    
    Args:
        grid: The navigation grid
        current: Current position as (x, y) cell coordinates
        goal: Goal position as (x, y) cell coordinates
        
    Returns:
        Estimated cost from current to goal
    """
    # Convert to world coordinates
    current_world = grid.cell_to_coords(*current)
    goal_world = grid.cell_to_coords(*goal)
    
    # Euclidean distance
    dx = goal_world[0] - current_world[0]
    dy = goal_world[1] - current_world[1]
    
    return np.sqrt(dx**2 + dy**2)


def reconstruct_path(
    came_from: Dict[Tuple[int, int], Tuple[int, int]],
    current: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Reconstruct the path from start to goal.
    
    Args:
        came_from: Dictionary mapping node to previous node
        current: Current (goal) node
        
    Returns:
        List of nodes forming the path
    """
    path = [current]
    
    while current in came_from:
        current = came_from[current]
        path.append(current)
    
    return list(reversed(path))


def generate_expanding_square_pattern(
    center: Tuple[int, int],
    max_distance: int,
    step_size: int = 1
) -> List[Tuple[int, int]]:
    """
    Generate an expanding square search pattern.
    
    Args:
        center: Center point of the pattern as (x, y) cell coordinates
        max_distance: Maximum distance from center in grid cells
        step_size: Step size for each leg of the pattern
        
    Returns:
        List of (x, y) cell coordinates forming the pattern
    """
    path = [center]
    
    cx, cy = center
    
    for d in range(step_size, max_distance + 1, step_size):
        # Top side (moving right)
        for x in range(cx - d + step_size, cx + d + 1, step_size):
            path.append((x, cy + d))
        
        # Right side (moving down)
        for y in range(cy + d - step_size, cy - d - 1, -step_size):
            path.append((cx + d, y))
        
        # Bottom side (moving left)
        for x in range(cx + d - step_size, cx - d - 1, -step_size):
            path.append((x, cy - d))
        
        # Left side (moving up)
        for y in range(cy - d + step_size, cy + d + 1, step_size):
            path.append((cx - d, y))
    
    return path


def generate_sector_search_pattern(
    center: Tuple[int, int],
    max_distance: int,
    num_triangles: int = 6,
    initial_orientation: float = 0.0,
    triangle_angle: float = np.pi/3  # 60 degrees
) -> List[Tuple[int, int]]:
    """
    Generate a sector search pattern following IAMSAR guidelines.
    
    The sector search pattern consists of triangular sweeps from a datum point,
    with each triangle being rotated to eventually cover a full circular area.
    
    Args:
        center: Center point (datum) of the pattern as (x, y) cell coordinates
        max_distance: Maximum distance from center in grid cells
        num_triangles: Number of triangular sweeps to cover the area
        initial_orientation: Initial orientation angle in radians
        triangle_angle: Angle of each triangular sector in radians (default 60°)
        
    Returns:
        List of (x, y) cell coordinates forming the pattern
    """
    path = [center]
    
    cx, cy = center
    
    # Calculate rotation angle between triangular sweeps
    rotation_step = 2 * np.pi / num_triangles
    
    # For each triangular sweep
    for sweep in range(num_triangles):
        sweep_orientation = initial_orientation + sweep * rotation_step
        
        # First leg - outbound on initial heading
        angle1 = sweep_orientation
        endpoint1_x = int(cx + max_distance * np.cos(angle1))
        endpoint1_y = int(cy + max_distance * np.sin(angle1))
        
        # Add points along the first leg
        path.extend(line_points(cx, cy, endpoint1_x, endpoint1_y))
        
        # Second leg - turn by triangle_angle and head back to center
        angle2 = angle1 + triangle_angle
        midpoint_x = int(cx + max_distance * np.cos(angle2))
        midpoint_y = int(cy + max_distance * np.sin(angle2))
        
        # Add points along the second leg
        path.extend(line_points(endpoint1_x, endpoint1_y, midpoint_x, midpoint_y))
        
        # Third leg - back to center
        path.extend(line_points(midpoint_x, midpoint_y, cx, cy))
    
    return path


def generate_parallel_search_pattern(
    start: Tuple[int, int],
    width: int,
    height: int,
    spacing: int,
    orientation: float = 0.0
) -> List[Tuple[int, int]]:
    """
    Generate a parallel search pattern.
    
    Args:
        start: Starting point as (x, y) cell coordinates
        width: Width of the area to search in grid cells
        height: Height of the area to search in grid cells
        spacing: Spacing between parallel tracks in grid cells
        orientation: Orientation angle in radians
        
    Returns:
        List of (x, y) cell coordinates forming the pattern
    """
    # Create a rectangle of the search area
    sx, sy = start
    
    # Calculate corner points of the rectangle
    cos_o = np.cos(orientation)
    sin_o = np.sin(orientation)
    
    # Calculate rectangle corners
    corners = [
        (sx, sy),
        (sx + width * cos_o, sy + width * sin_o),
        (sx + width * cos_o - height * sin_o, sy + width * sin_o + height * cos_o),
        (sx - height * sin_o, sy + height * cos_o)
    ]
    
    # Convert to integer coordinates
    corners = [(int(x), int(y)) for x, y in corners]
    
    # Generate the path
    path = []
    
    # Add first leg
    path.extend(line_points(*corners[0], *corners[1]))
    
    # Number of legs
    num_legs = height // spacing + 1
    
    # Direction of movement (alternating)
    direction = 1
    
    for i in range(num_legs):
        # Calculate track endpoints
        track_start_x = corners[0][0] + i * spacing * (-sin_o)
        track_start_y = corners[0][1] + i * spacing * cos_o
        track_end_x = corners[1][0] + i * spacing * (-sin_o)
        track_end_y = corners[1][1] + i * spacing * cos_o
        
        # Convert to integers
        start_x, start_y = int(track_start_x), int(track_start_y)
        end_x, end_y = int(track_end_x), int(track_end_y)
        
        # Add this track
        if direction > 0:
            path.extend(line_points(start_x, start_y, end_x, end_y))
        else:
            path.extend(line_points(end_x, end_y, start_x, start_y))
        
        # Flip direction for next track
        direction *= -1
    
    return path


def line_points(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Generate a list of cell coordinates along a line using Bresenham's algorithm.
    
    Args:
        x0, y0: Starting coordinates
        x1, y1: Ending coordinates
        
    Returns:
        List of (x, y) cell coordinates along the line
    """
    points = []
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        points.append((x0, y0))
        
        if x0 == x1 and y0 == y1:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            if x0 == x1:
                break
            err -= dy
            x0 += sx
        if e2 < dx:
            if y0 == y1:
                break
            err += dx
            y0 += sy
    
    return points