"""
Path planning algorithms for USV navigation with currents.
"""
from typing import List, Tuple, Dict, Set, Optional, Any, TYPE_CHECKING
import heapq
import numpy as np
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import split

# This prevents circular imports while maintaining type checking
if TYPE_CHECKING:
    from ..core.grid import NavigationGrid
else:
    # For runtime
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
    step_size: int = 1,
    grid: Optional['NavigationGrid'] = None
) -> List[Tuple[int, int]]:
    """
    Generate an expanding square search pattern following IAMSAR guidelines.
    
    This implements the correct spiral pattern with leg lengths that increase
    by step_size for each new leg, following the pattern:
    - First leg: length d (to the right)
    - Second leg: length d (upward)
    - Third leg: length 2d (to the left)
    - Fourth leg: length 2d (downward)
    - Fifth leg: length 3d (to the right)
    - etc.
    
    Args:
        center: Center point of the pattern as (x, y) cell coordinates
        max_distance: Maximum distance from center in grid cells
        step_size: Step size for each leg (d in the IAMSAR pattern)
        grid: Optional NavigationGrid for obstacle avoidance
        
    Returns:
        List of (x, y) cell coordinates forming the pattern
    """
    # Validate center point
    cx, cy = center
    
    if grid and grid.is_obstacle(cx, cy):
        # If center is on obstacle, find a nearby safe point
        safe_center = find_closest_non_obstacle(center, grid, max_distance=5)
        if safe_center:
            center = safe_center
            cx, cy = center
        else:
            # Can't find a safe center point
            return []
    
    # Start with center point
    path = [center]
    current_pos = center
    
    # Directions: right, up, left, down
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    
    # Initial leg length
    leg_length = step_size
    
    # Current direction
    direction = 0  # start going right
    
    # Keep track of legs completed
    legs_completed = 0
    
    # Continue until reaching max distance
    while leg_length <= max_distance:
        # Calculate the end point of this leg
        end_x = current_pos[0] + leg_length * dx[direction]
        end_y = current_pos[1] + leg_length * dy[direction]
        end_point = (end_x, end_y)
        
        # Generate points along this leg
        leg_points = line_points(current_pos[0], current_pos[1], end_x, end_y)
        
        # Skip the first point to avoid duplication (except for first leg)
        if legs_completed > 0:
            leg_points = leg_points[1:]
        
        # Apply obstacle avoidance
        if grid is not None:
            safe_leg = avoid_obstacles_on_leg(leg_points, grid)
            
            # If the leg was modified due to obstacles, we need to update the end point
            if safe_leg:
                path.extend(safe_leg)
                current_pos = safe_leg[-1] if safe_leg else current_pos
            else:
                current_pos = end_point
        else:
            path.extend(leg_points)
            current_pos = end_point
        
        # Move to next leg
        legs_completed += 1
        
        # Update direction
        direction = (direction + 1) % 4
        
        # Update leg length every 2 legs according to IAMSAR pattern
        if legs_completed % 2 == 0:
            leg_length += step_size
    
    return path


def generate_sector_search_pattern(
    center: Tuple[int, int],
    max_distance: int,
    num_triangles: int = 6,
    initial_orientation: float = 0.0,
    triangle_angle: float = np.pi/3,  # 60 degrees
    grid: Optional['NavigationGrid'] = None
) -> List[Tuple[int, int]]:
    """
    Generate a sector search pattern following IAMSAR guidelines with obstacle avoidance.
    
    The sector search pattern consists of triangular sweeps from a datum point,
    with each triangle being rotated to eventually cover a full circular area.
    
    Args:
        center: Center point (datum) of the pattern as (x, y) cell coordinates
        max_distance: Maximum distance from center in grid cells
        num_triangles: Number of triangular sweeps to cover the area
        initial_orientation: Initial orientation angle in radians
        triangle_angle: Angle of each triangular sector in radians (default 60°)
        grid: Optional NavigationGrid for obstacle avoidance
        
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
        
        # Generate points along the first leg
        leg1_points = line_points(cx, cy, endpoint1_x, endpoint1_y)
        
        # Apply obstacle avoidance to this leg
        if grid is not None:
            path.extend(avoid_obstacles_on_leg(leg1_points, grid))
        else:
            path.extend(leg1_points)
        
        # Second leg - turn by triangle_angle and head back to center
        angle2 = angle1 + triangle_angle
        midpoint_x = int(cx + max_distance * np.cos(angle2))
        midpoint_y = int(cy + max_distance * np.sin(angle2))
        
        # Generate points along the second leg
        leg2_points = line_points(endpoint1_x, endpoint1_y, midpoint_x, midpoint_y)
        
        # Apply obstacle avoidance to this leg
        if grid is not None:
            path.extend(avoid_obstacles_on_leg(leg2_points, grid))
        else:
            path.extend(leg2_points)
        
        # Third leg - back to center
        leg3_points = line_points(midpoint_x, midpoint_y, cx, cy)
        
        # Apply obstacle avoidance to this leg
        if grid is not None:
            path.extend(avoid_obstacles_on_leg(leg3_points, grid))
        else:
            path.extend(leg3_points)
    
    return path


def generate_parallel_search_pattern(
    start: Tuple[int, int],
    width: int,
    height: int,
    spacing: int,
    orientation: float = 0.0,
    grid: Optional['NavigationGrid'] = None
) -> List[Tuple[int, int]]:
    """
    Generate a parallel search pattern with proper obstacle avoidance using
    geometric operations to ensure complete coverage of the search area.
    
    This approach:
    1. Creates a polygon representing the search area
    2. Subtracts obstacle polygons to create a valid search region
    3. Generates parallel tracks within that region
    4. Connects tracks while avoiding obstacles
    
    Args:
        start: Starting point as (x, y) cell coordinates
        width: Width of the area to search in grid cells
        height: Height of the area to search in grid cells
        spacing: Spacing between parallel tracks in grid cells
        orientation: Orientation angle in radians
        grid: Optional NavigationGrid for obstacle avoidance
        
    Returns:
        List of (x, y) cell coordinates forming the pattern
    """
    # Check if start point is valid
    sx, sy = start
    if grid and grid.is_obstacle(sx, sy):
        # If start is on obstacle, find a nearby safe point
        safe_start = find_closest_non_obstacle(start, grid, max_distance=5)
        if safe_start:
            start = safe_start
            sx, sy = start
        else:
            # Can't find a safe start point
            return []
    
    # Calculate unit vectors for track direction and spacing
    cos_o = np.cos(orientation)
    sin_o = np.sin(orientation)
    
    # Create the four corners of the search area
    corners = [
        (sx, sy),  # bottom-left
        (sx + width * cos_o - height * sin_o, sy + width * sin_o + height * cos_o),  # top-left
        (sx + width * cos_o, sy + width * sin_o),  # bottom-right
        (sx + width * cos_o + height * sin_o, sy + width * sin_o - height * cos_o)  # top-right
    ]
    
    # Create a polygon representing the search area
    search_area = Polygon(corners)
    
    # If we have a grid with obstacles, create obstacle polygons and subtract them
    search_region = search_area
    obstacle_polygons = []
    
    if grid is not None:
        # Find all obstacle cells in the grid
        obstacle_cells = []
        
        # Define a bounding box around the search area to limit our search
        min_x = min(p[0] for p in corners)
        max_x = max(p[0] for p in corners)
        min_y = min(p[1] for p in corners)
        max_y = max(p[1] for p in corners)
        
        # Add some margin
        margin = max(width, height) // 5  # Increase margin to ensure we catch all obstacles
        min_x = max(0, int(min_x - margin))
        min_y = max(0, int(min_y - margin))
        max_x = min(grid.grid_size[0] - 1, int(max_x + margin))
        max_y = min(grid.grid_size[1] - 1, int(max_y + margin))
        
        # Create a polygon for the search area with a small buffer
        search_polygon = search_area.buffer(grid.cell_size * 2)
        
        # Find all obstacle cells within the bounding box that intersect with the search area
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if grid.is_obstacle(x, y):
                    # Convert to world coordinates
                    world_x, world_y = grid.cell_to_coords(x, y)
                    # Check if this obstacle cell intersects with the search area
                    cell_point = Point(world_x, world_y)
                    if search_polygon.contains(cell_point) or search_polygon.intersects(cell_point):
                        obstacle_cells.append((world_x, world_y))
        
        # Group adjacent obstacle cells into polygons
        if obstacle_cells:
            # Create obstacle polygons with a small buffer to ensure separation from tracks
            cell_size = grid.cell_size * 1.2  # Add a small buffer
            for ox, oy in obstacle_cells:
                # Create a square polygon for this obstacle cell
                obstacle_poly = Polygon([
                    (ox - cell_size/2, oy - cell_size/2),
                    (ox + cell_size/2, oy - cell_size/2),
                    (ox + cell_size/2, oy + cell_size/2),
                    (ox - cell_size/2, oy + cell_size/2)
                ])
                obstacle_polygons.append(obstacle_poly)
            
            # Combine all obstacle polygons
            if obstacle_polygons:
                combined_obstacles = obstacle_polygons[0]
                for poly in obstacle_polygons[1:]:
                    combined_obstacles = combined_obstacles.union(poly)
                
                # Subtract obstacles from search area
                search_region = search_area.difference(combined_obstacles)
    
    # Generate parallel track lines that cover the search region
    path = []
    
    # Calculate the track direction and spacing unit vectors
    track_dx = cos_o
    track_dy = sin_o
    spacing_dx = -sin_o
    spacing_dy = cos_o
    
    # Calculate number of tracks needed
    num_tracks = height // spacing + 1
    
    # Direction of travel (alternates)
    direction = 1
    
    # Previous track end point (for connecting tracks)
    prev_end = None
    
    # Generate all tracks
    for i in range(num_tracks):
        # Calculate offset from start
        offset = i * spacing
        
        # Calculate track start and end points
        track_start_x = sx + offset * spacing_dx
        track_start_y = sy + offset * spacing_dy
        track_end_x = track_start_x + width * track_dx * 1.2  # Extend slightly to ensure coverage
        track_end_y = track_start_y + width * track_dy * 1.2
        
        # Create a line representing this track
        if direction > 0:
            track_line = LineString([(track_start_x, track_start_y), (track_end_x, track_end_y)])
        else:
            track_line = LineString([(track_end_x, track_end_y), (track_start_x, track_start_y)])
        
        # Intersect with the search region to get valid track segments
        if isinstance(search_region, MultiPolygon):
            # Handle multiple polygons
            track_segments = []
            for poly in search_region.geoms:
                if track_line.intersects(poly):
                    intersection = track_line.intersection(poly)
                    if not intersection.is_empty:
                        if isinstance(intersection, LineString):
                            track_segments.append(intersection)
                        elif hasattr(intersection, 'geoms'):  # MultiLineString
                            track_segments.extend([ls for ls in intersection.geoms if isinstance(ls, LineString)])
        else:
            # Single polygon
            intersection = track_line.intersection(search_region)
            track_segments = []
            if not intersection.is_empty:
                if isinstance(intersection, LineString):
                    track_segments.append(intersection)
                elif hasattr(intersection, 'geoms'):  # MultiLineString
                    track_segments.extend([ls for ls in intersection.geoms if isinstance(ls, LineString)])
        
        # Convert track segments to point lists and add to path
        track_points = []
        for segment in track_segments:
            # Get coordinates and convert to integer grid coordinates
            coords = list(segment.coords)
            segment_points = [(int(x), int(y)) for x, y in coords]
            
            # If we have multiple points, densify the line to ensure no gaps
            if len(segment_points) > 1:
                dense_points = []
                for j in range(len(segment_points) - 1):
                    p1 = segment_points[j]
                    p2 = segment_points[j + 1]
                    dense_points.extend(line_points(p1[0], p1[1], p2[0], p2[1]))
                segment_points = dense_points
            
            # Filter out any points that end up on obstacles (can happen due to rounding)
            if grid is not None:
                segment_points = [p for p in segment_points if not grid.is_obstacle(p[0], p[1])]
            
            track_points.extend(segment_points)
        
        # Connect from previous track if needed
        if i > 0 and prev_end is not None and track_points:
            # Find the closest point on the current track to connect to
            closest_idx = 0
            min_dist = float('inf')
            for j, point in enumerate(track_points):
                dist = (point[0] - prev_end[0])**2 + (point[1] - prev_end[1])**2
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = j
            
            # Create a connector between tracks
            connector_points = line_points(prev_end[0], prev_end[1], 
                                          track_points[closest_idx][0], track_points[closest_idx][1])
            
            # Skip the first point to avoid duplication
            if connector_points:
                connector_points = connector_points[1:]
            
            # Apply obstacle avoidance to connector if needed
            if grid is not None and connector_points:
                connector_points = avoid_obstacles_on_leg(connector_points, grid)
                
                # Double-check that no points are on obstacles (can happen due to rounding or approximation)
                connector_points = [p for p in connector_points if not grid.is_obstacle(p[0], p[1])]
            
            # Add connector to path
            path.extend(connector_points)
            
            # Reorder track points to start from the connection point
            track_points = track_points[closest_idx:] + track_points[:closest_idx]
        
        # Add track points to the overall path
        if track_points:
            path.extend(track_points)
            prev_end = track_points[-1]
        
        # Flip direction for next track
        direction *= -1
    
    return path


def avoid_obstacles_on_leg(
    leg_points: List[Tuple[int, int]], 
    grid: 'NavigationGrid'
) -> List[Tuple[int, int]]:
    """
    Apply obstacle avoidance to a leg of a search pattern.
    
    This function attempts to maintain the general shape of the pattern
    by shifting legs parallel to their original direction when obstacles
    are encountered, rather than finding totally new paths.
    
    Args:
        leg_points: List of points forming the original leg
        grid: NavigationGrid containing obstacle information
        
    Returns:
        Modified list of points that avoids obstacles while maintaining pattern shape
    """
    if not leg_points:
        return []
    
    # If no obstacles on this leg, return the original points
    if all(not grid.is_obstacle(x, y) for x, y in leg_points):
        return leg_points
    
    # Get start and end points
    start_point = leg_points[0]
    end_point = leg_points[-1]
    
    # Calculate the direction of the leg
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    
    # If just a point, not a line
    if dx == 0 and dy == 0:
        if not grid.is_obstacle(*start_point):
            return [start_point]
        else:
            new_point = find_closest_non_obstacle(start_point, grid, max_distance=3)
            return [new_point] if new_point else []
    
    # Normalize to get direction vector
    length = np.sqrt(dx**2 + dy**2)
    direction_x = dx / length
    direction_y = dy / length
    
    # Get perpendicular direction (for offsetting)
    perp_x = -direction_y
    perp_y = direction_x
    
    # Try offsets in alternating directions
    max_offset = 10  # Maximum distance to shift the leg
    
    # Check if original points are valid
    if not grid.is_obstacle(*start_point) and not grid.is_obstacle(*end_point):
        # Try to follow original leg with modest deviations
        valid_points = []
        for point in leg_points:
            if grid.is_obstacle(*point):
                # Find nearest safe point perpendicular to leg direction
                found = False
                for offset in range(1, max_offset + 1):
                    # Try both sides perpendicular to the leg
                    for side in [1, -1]:
                        new_x = int(point[0] + side * offset * perp_x)
                        new_y = int(point[1] + side * offset * perp_y)
                        
                        if (0 <= new_x < grid.grid_size[0] and 
                            0 <= new_y < grid.grid_size[1] and 
                            not grid.is_obstacle(new_x, new_y)):
                            valid_points.append((new_x, new_y))
                            found = True
                            break
                    if found:
                        break
                if not found:
                    # If no safe point found perpendicular, try moving along the leg
                    safe_point = find_closest_non_obstacle(point, grid, max_distance=3)
                    if safe_point:
                        valid_points.append(safe_point)
            else:
                valid_points.append(point)
        
        if valid_points:
            return valid_points
    
    # If we can't maintain the leg's shape, try shifting the entire leg
    for offset in range(1, max_offset + 1):
        for side in [1, -1]:
            # Calculate offset points
            offset_start_x = int(start_point[0] + side * offset * perp_x)
            offset_start_y = int(start_point[1] + side * offset * perp_y)
            offset_end_x = int(end_point[0] + side * offset * perp_x)
            offset_end_y = int(end_point[1] + side * offset * perp_y)
            
            # Check if offset points are valid
            if (0 <= offset_start_x < grid.grid_size[0] and 
                0 <= offset_start_y < grid.grid_size[1] and
                0 <= offset_end_x < grid.grid_size[0] and 
                0 <= offset_end_y < grid.grid_size[1] and
                not grid.is_obstacle(offset_start_x, offset_start_y) and
                not grid.is_obstacle(offset_end_x, offset_end_y)):
                
                # Generate offset leg points
                offset_points = line_points(offset_start_x, offset_start_y, 
                                           offset_end_x, offset_end_y)
                
                # Check if the offset line avoids obstacles
                if not any(grid.is_obstacle(x, y) for x, y in offset_points):
                    return offset_points
    
    # If we still can't find a good path, use A* as last resort
    start_point = find_closest_non_obstacle(start_point, grid, max_distance=5) or start_point
    end_point = find_closest_non_obstacle(end_point, grid, max_distance=5) or end_point
    
    if not grid.is_obstacle(*start_point) and not grid.is_obstacle(*end_point):
        path = a_star_current_aware(grid, start_point, end_point)
        if path:
            return path
    
    # Last resort: return any valid points from the original leg
    valid_points = []
    for p in leg_points:
        if not grid.is_obstacle(*p):
            valid_points.append(p)
    return valid_points


def find_closest_non_obstacle(
    point: Tuple[int, int], 
    grid: 'NavigationGrid',
    max_distance: int = 3
) -> Optional[Tuple[int, int]]:
    """
    Find the closest non-obstacle point to the given point.
    
    Args:
        point: The reference point (x, y)
        grid: NavigationGrid containing obstacle information
        max_distance: Maximum distance to search
        
    Returns:
        Closest non-obstacle point, or None if none found
    """
    x, y = point
    
    # Check spiral pattern around the point
    for d in range(1, max_distance + 1):
        # Check top and bottom rows
        for dx in range(-d, d + 1):
            if not grid.is_obstacle(x + dx, y + d):
                return (x + dx, y + d)
            if not grid.is_obstacle(x + dx, y - d):
                return (x + dx, y - d)
        
        # Check left and right columns
        for dy in range(-d + 1, d):
            if not grid.is_obstacle(x + d, y + dy):
                return (x + d, y + dy)
            if not grid.is_obstacle(x - d, y + dy):
                return (x - d, y + dy)
    
    return None


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