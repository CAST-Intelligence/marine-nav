"""
Spatio-Temporal A* Path Planning for USV navigation in time-varying environments.
"""
from typing import List, Tuple, Dict, Set, Optional, Any, Union
import heapq
import numpy as np
from ..core.temporal_grid import TemporalNavigationGrid


class STAStarNode:
    """Helper class to represent nodes in the ST-A* search."""
    
    def __init__(self, x: int, y: int, time_idx: float):
        self.x = x
        self.y = y
        self.time_idx = time_idx
        # Using __slots__ for memory efficiency (crucial for large-scale path planning)
        
    @property
    def coords(self) -> Tuple[int, int]:
        return (self.x, self.y)
        
    def __eq__(self, other):
        if not isinstance(other, STAStarNode):
            return False
        return (self.x == other.x and self.y == other.y and 
                abs(self.time_idx - other.time_idx) < 0.001)
    
    def __hash__(self):
        # For time values, we discretize to allow for proper hashing
        # This creates a trade-off between time precision and memory/performance
        discrete_time = int(self.time_idx * 100)  # 100x multiplier gives 0.01 time step precision
        return hash((self.x, self.y, discrete_time))
    
    def __lt__(self, other):
        # For priority queue comparison, arbitrary but stable ordering
        if self.time_idx != other.time_idx:
            return self.time_idx < other.time_idx
        if self.x != other.x:
            return self.x < other.x
        return self.y < other.y


def spatio_temporal_astar(
    grid: TemporalNavigationGrid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    start_time_idx: float = 0.0,
    usv_cruise_speed: float = 1.0,
    usv_max_speed: float = 1.5,
    optimization_mode: str = 'fastest_time',
    heuristic_weight: float = 1.0,
    max_energy: Optional[float] = None,
    max_time_idx: Optional[float] = None
) -> Dict[str, Any]:
    """
    Spatio-Temporal A* algorithm for path planning with time-varying environments.
    
    ST-A* incorporates time as a dimension in the search space, allowing accurate
    planning in dynamic environments. The algorithm handles time-dependent costs,
    energy constraints, and time horizon limitations.
    
    Args:
        grid: The temporal navigation grid with current/weather information
        start: Starting point as (x, y) cell coordinates
        goal: Goal point as (x, y) cell coordinates
        start_time_idx: Starting time index in the forecast
        usv_cruise_speed: The cruise speed of the USV in m/s
        usv_max_speed: The maximum speed of the USV in m/s
        optimization_mode: 'fastest_time' or 'lowest_energy'
        heuristic_weight: Weight for the heuristic (1.0 = standard A*)
        max_energy: Optional energy capacity constraint
        max_time_idx: Optional time horizon limit
        
    Returns:
        Dictionary containing:
            'path': List of (x, y, time_index) points
            'energy': Total energy consumption
            'time': Total time duration
            'power_profile': List of power settings per segment
            'feasible': Whether the path satisfies all constraints
    """
    # Initialize start and goal nodes
    start_node = STAStarNode(start[0], start[1], start_time_idx)
    
    # Open list (priority queue) and closed set
    open_list = []
    closed_set = set()
    
    # Tracking dictionaries for g-scores, energy and path reconstruction
    g_cost = {start_node: 0}
    g_energy = {start_node: 0}
    g_time = {start_node: 0}
    came_from = {}
    
    # Initial heuristic score for start node
    h_cost = heuristic(grid, start, goal, optimization_mode, usv_cruise_speed)
    f_cost = h_cost * heuristic_weight
    
    # Push start node to open list with initial f-score
    heapq.heappush(open_list, (f_cost, start_node))
    
    # Track the best goal node found and its cost
    best_goal_node = None
    best_goal_cost = float('inf')
    
    # Main A* loop
    while open_list:
        # Pop the node with lowest f-cost
        _, current = heapq.heappop(open_list)
        
        # Check if this is a goal state
        if current.coords == goal:
            # Update best goal if this one is better
            if g_cost[current] < best_goal_cost:
                best_goal_node = current
                best_goal_cost = g_cost[current]
            # We can optionally continue searching for better paths,
            # but for this implementation we'll terminate on first goal found
            break
        
        # Skip if already evaluated with better cost
        if current in closed_set:
            continue
            
        # Add to closed set
        closed_set.add(current)
        
        # Check time horizon constraint
        if max_time_idx is not None and current.time_idx >= max_time_idx:
            continue
        
        # Expand neighbors
        for neighbor_coords in grid.get_neighbors(current.x, current.y):
            nx, ny = neighbor_coords
            
            # Calculate edge cost accounting for time-varying currents
            edge_details = grid.calculate_travel_cost_time(
                current.x, current.y, nx, ny, current.time_idx,
                usv_cruise_speed, usv_max_speed, optimization_mode,
                return_details=True
            )
            
            # Skip if edge is invalid (infinite cost)
            if edge_details['cost'] == float('inf'):
                continue
                
            # Create neighbor node with updated time
            neighbor = STAStarNode(nx, ny, edge_details['time_idx_end'])
            
            # Calculate tentative g-score for this path
            tentative_g_cost = g_cost[current] + edge_details['cost']
            tentative_g_energy = g_energy[current] + edge_details['energy']
            tentative_g_time = g_time[current] + edge_details['time']
            
            # Check energy constraint
            if max_energy is not None and tentative_g_energy > max_energy:
                continue
            
            # Update if this is a better path to neighbor
            if (neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]):
                # Update trackers
                came_from[neighbor] = (current, edge_details)
                g_cost[neighbor] = tentative_g_cost
                g_energy[neighbor] = tentative_g_energy
                g_time[neighbor] = tentative_g_time
                
                # Calculate heuristic for neighbor
                h_cost = heuristic(grid, (nx, ny), goal, optimization_mode, usv_cruise_speed)
                f_cost = tentative_g_cost + h_cost * heuristic_weight
                
                # Add to open list
                heapq.heappush(open_list, (f_cost, neighbor))
    
    # If we found a valid goal state, reconstruct the path
    if best_goal_node is not None:
        path = []
        power_profile = []
        current = best_goal_node
        
        while current in came_from:
            # Add the current node to the path
            path.append((current.x, current.y, current.time_idx))
            
            # Get edge details and previous node
            prev, edge_details = came_from[current]
            
            # Calculate power setting based on water speed
            if 'drift' in edge_details and edge_details['drift']:
                power = 0
            else:
                usv_vel = edge_details['usv_vel']
                usv_water_speed = np.linalg.norm(usv_vel)
                power = int(100 * usv_water_speed / usv_max_speed)
                
            power_profile.append(power)
            current = prev
            
        # Add start node
        path.append((start_node.x, start_node.y, start_node.time_idx))
        
        # Reverse the lists to get start-to-goal order
        path.reverse()
        power_profile.reverse()
        
        return {
            'path': path,
            'energy': g_energy[best_goal_node],
            'time': g_time[best_goal_node],
            'power_profile': power_profile,
            'feasible': True
        }
    
    # No path found
    return {
        'path': [],
        'energy': 0,
        'time': 0,
        'power_profile': [],
        'feasible': False
    }


def heuristic(
    grid: TemporalNavigationGrid,
    current: Tuple[int, int],
    goal: Tuple[int, int],
    optimization_mode: str,
    usv_speed: float
) -> float:
    """
    Calculate heuristic for ST-A*.
    
    For fastest time: Euclidean distance / USV speed
    For lowest energy: Energy to travel at cruise speed
    
    Args:
        grid: The navigation grid
        current: Current position as (x, y) cell coordinates
        goal: Goal position as (x, y) cell coordinates
        optimization_mode: 'fastest_time' or 'lowest_energy'
        usv_speed: The cruise speed of the USV in m/s
        
    Returns:
        Estimated cost from current to goal
    """
    # Convert to world coordinates
    current_world = grid.cell_to_coords(*current)
    goal_world = grid.cell_to_coords(*goal)
    
    # Euclidean distance
    dx = goal_world[0] - current_world[0]
    dy = goal_world[1] - current_world[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    if optimization_mode == 'fastest_time':
        # Optimistic time estimate (straight line at cruise speed)
        return distance / usv_speed
    else:  # 'lowest_energy'
        # Simple energy model, could be more sophisticated
        # Using cubic relationship to speed as in the cost function
        time_estimate = distance / usv_speed
        return time_estimate


def find_multi_segment_path(
    grid: TemporalNavigationGrid,
    waypoints: List[Tuple[int, int]],
    start_time_idx: float = 0.0,
    usv_cruise_speed: float = 1.0,
    usv_max_speed: float = 1.5,
    optimization_mode: str = 'fastest_time',
    max_energy: Optional[float] = None,
    max_carrying_capacity: Optional[float] = None,
    pickup_dropoff_amounts: Optional[Dict[int, float]] = None
) -> Dict[str, Any]:
    """
    Find a path through multiple waypoints with ST-A*, checking all constraints.
    
    Args:
        grid: The temporal navigation grid
        waypoints: List of (x, y) cell coordinates to visit in order
        start_time_idx: Starting time index
        usv_cruise_speed: The cruise speed of the USV in m/s
        usv_max_speed: The maximum speed of the USV in m/s
        optimization_mode: 'fastest_time' or 'lowest_energy'
        max_energy: Optional energy capacity constraint
        max_carrying_capacity: Optional carrying capacity constraint
        pickup_dropoff_amounts: Dict mapping waypoint index to +/- load amount
        
    Returns:
        Dictionary with results for the entire mission
    """
    if len(waypoints) < 2:
        return {
            'path': [],
            'energy': 0,
            'time': 0,
            'power_profile': [],
            'feasible': False,
            'error': 'Need at least start and goal waypoints'
        }
    
    # Initialize result containers
    complete_path = []
    complete_power_profile = []
    total_energy = 0
    total_time = 0
    current_time_idx = start_time_idx
    current_load = 0
    
    # Track constraint violations
    energy_exceeded = False
    capacity_exceeded = False
    
    # Process each segment
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        goal = waypoints[i+1]
        
        # Update load if pickup/dropoff info provided
        if pickup_dropoff_amounts and i in pickup_dropoff_amounts:
            current_load += pickup_dropoff_amounts[i]
            if max_carrying_capacity is not None and current_load > max_carrying_capacity:
                capacity_exceeded = True
        
        # Calculate remaining energy
        remaining_energy = None
        if max_energy is not None:
            remaining_energy = max_energy - total_energy
            if remaining_energy <= 0:
                energy_exceeded = True
                break
        
        # Find path for this segment
        result = spatio_temporal_astar(
            grid, start, goal, current_time_idx,
            usv_cruise_speed, usv_max_speed, optimization_mode,
            max_energy=remaining_energy
        )
        
        # If no path found, mark as infeasible
        if not result['feasible'] or not result['path']:
            return {
                'path': complete_path,
                'energy': total_energy,
                'time': total_time,
                'power_profile': complete_power_profile,
                'feasible': False,
                'error': f'No path found for segment {i} ({start} to {goal})'
            }
        
        # Update running totals
        segment_path = result['path']
        
        # For second segment onwards, skip the first point to avoid duplication
        if i > 0 and segment_path:
            segment_path = segment_path[1:]
            
        complete_path.extend(segment_path)
        complete_power_profile.extend(result['power_profile'])
        total_energy += result['energy']
        total_time += result['time']
        
        # Update current time for next segment
        if segment_path:
            current_time_idx = segment_path[-1][2]  # Time index of last point
    
    # Final load update for last waypoint
    if pickup_dropoff_amounts and (len(waypoints) - 1) in pickup_dropoff_amounts:
        current_load += pickup_dropoff_amounts[len(waypoints) - 1]
        if max_carrying_capacity is not None and current_load > max_carrying_capacity:
            capacity_exceeded = True
    
    # Check final constraints
    feasible = not (energy_exceeded or capacity_exceeded)
    error_msg = None
    if not feasible:
        if energy_exceeded:
            error_msg = 'Energy capacity exceeded'
        elif capacity_exceeded:
            error_msg = 'Carrying capacity exceeded'
    
    return {
        'path': complete_path,
        'energy': total_energy,
        'time': total_time,
        'power_profile': complete_power_profile,
        'feasible': feasible,
        'error': error_msg
    }


def analyze_drift_opportunities(
    grid: TemporalNavigationGrid,
    path: List[Tuple[int, int, float]],
    power_profile: List[int],
    drift_verification: bool = True
) -> List[int]:
    """
    Analyze a path for drift opportunities and verify them with simulation.
    
    Args:
        grid: The temporal navigation grid
        path: List of (x, y, time_idx) points along the path
        power_profile: List of power settings per segment
        drift_verification: Whether to verify drift paths with simulation
        
    Returns:
        List of segment indices that are drift opportunities
    """
    drift_segments = []
    
    # Identify zero-power segments directly from power profile
    for i, power in enumerate(power_profile):
        if power == 0:
            drift_segments.append(i)
    
    # If drift verification requested and we have the environment field
    if drift_verification and grid.environment_field and len(path) > 1:
        verified_segments = []
        
        for i in drift_segments:
            # Get segment endpoints
            start_x, start_y, start_time = path[i]
            end_x, end_y, end_time = path[i+1]
            
            # Convert to world coordinates
            start_world_x, start_world_y = grid.cell_to_coords(start_x, start_y)
            end_world_x, end_world_y = grid.cell_to_coords(end_x, end_y)
            
            # Calculate drift duration in seconds
            drift_duration = (end_time - start_time) * grid.time_step_duration
            
            # Simulate drift path
            drift_path = grid.environment_field.integrate_drift_path(
                start_world_x, start_world_y, start_time, drift_duration
            )
            
            # Check if simulated endpoint is close to target endpoint
            if drift_path:
                final_x, final_y, _ = drift_path[-1]
                distance_to_target = np.sqrt(
                    (final_x - end_world_x)**2 + 
                    (final_y - end_world_y)**2
                )
                
                # Set acceptance radius based on cell size
                acceptance_radius = grid.cell_size * 1.5
                
                if distance_to_target <= acceptance_radius:
                    verified_segments.append(i)
        
        return verified_segments
    
    return drift_segments