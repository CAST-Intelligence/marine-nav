"""
Grid Network Path Finding with CP-SAT
====================================

This example demonstrates how to model a grid-based network with 8-directional
movement for path planning using CP-SAT in Google OR-Tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from ortools.sat.python import cp_model
import math
import time

# --- Define Grid Parameters ---
GRID_SIZE = 10  # 10x10 grid
START_POINT = (0, 0)  # Top-left corner
END_POINT = (9, 9)  # Bottom-right corner

# --- Add Required Visit Sites ---
# These are locations that must be visited (like a TSP/VRP problem)
REQUIRED_SITES = [
    (2, 7),  # Example site 1
    (7, 2),  # Example site 2
    (4, 5),  # Example site 3
]

# Set to True to enforce visiting all required sites, False for simple path finding
MUST_VISIT_SITES = True

# --- Create Grid with Obstacles ---
# 0 = free cell, 1 = obstacle
np.random.seed(42)  # For reproducible results
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# Add random obstacles (less obstacles for easier path finding - 15% of grid)
num_obstacles = int(0.15 * GRID_SIZE * GRID_SIZE)
obstacle_indices = np.random.choice(
    GRID_SIZE * GRID_SIZE, 
    size=num_obstacles, 
    replace=False
)
for idx in obstacle_indices:
    row = idx // GRID_SIZE
    col = idx % GRID_SIZE
    # Don't place obstacles at start, end, or required sites
    location = (row, col)
    if (location != START_POINT and 
        location != END_POINT and
        location not in REQUIRED_SITES):
        grid[row, col] = 1
        
# Create a path directly from START to END to ensure feasibility
# This is a simple manhattan path that creates a valid diagonal
# We'll clear obstacles along this path
direct_path = []
r, c = START_POINT
target_r, target_c = END_POINT

# Create a straight diagonal path as much as possible
while r != target_r or c != target_c:
    direct_path.append((r, c))
    if r < target_r:
        r += 1
    elif r > target_r:
        r -= 1
    
    if c < target_c:
        c += 1
    elif c > target_c:
        c -= 1

# Add endpoints
direct_path.append(END_POINT)

# Clear obstacles along this path
for r, c in direct_path:
    grid[r, c] = 0  # Clear any obstacles

# Create a separate grid to show site types for visualization
site_type_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
site_type_grid[START_POINT] = 1  # Start point
site_type_grid[END_POINT] = 2    # End point
for site in REQUIRED_SITES:
    site_type_grid[site] = 3     # Required visit site

# --- Time-dependent Movement Costs ---
# Define different costs for different time periods - more pronounced differences
time_periods = 3
time_costs = np.ones((time_periods, GRID_SIZE, GRID_SIZE)) 

# First period: Higher costs in top-right
time_costs[0, :GRID_SIZE//2, GRID_SIZE//2:] = 3.0

# Second period: Higher costs in bottom-left
time_costs[1, GRID_SIZE//2:, :GRID_SIZE//2] = 3.0

# Third period: Higher costs along diagonal
for i in range(GRID_SIZE):
    time_costs[2, i, i] = 2.5
    
# Set a base cost for all cells to ensure movement costs something
time_costs += 1.0  # All movements cost at least 1.0

# Define time interval information (actual values will be set after MAX_TIME is defined)
time_interval_info = {
    0: {"name": "Period 0", "speed_factor": 1.0},
    1: {"name": "Period 1", "speed_factor": 1.0},
    2: {"name": "Period 2", "speed_factor": 1.0}
}

# --- Movement Directions (8-directional) ---
# Format: (row_delta, col_delta)
DIRECTIONS = [
    (-1, 0),  # North
    (-1, 1),  # Northeast
    (0, 1),   # East
    (1, 1),   # Southeast
    (1, 0),   # South
    (1, -1),  # Southwest
    (0, -1),  # West
    (-1, -1)  # Northwest
]

# --- Network Construction (as adjacency list) ---
network = {}  # Format: network[(row, col)] = list of (next_row, next_col, cost)

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        if grid[row, col] == 1:
            continue  # Skip obstacles
        
        node = (row, col)
        network[node] = []
        
        for dr, dc in DIRECTIONS:
            new_row, new_col = row + dr, col + dc
            
            # Check if the new position is within bounds and not an obstacle
            if (0 <= new_row < GRID_SIZE and 
                0 <= new_col < GRID_SIZE and 
                grid[new_row, new_col] == 0):
                
                # Calculate base movement cost (1.0 for cardinal, 1.414 for diagonal)
                base_cost = 1.0
                if dr != 0 and dc != 0:  # Diagonal
                    base_cost = 1.414
                
                # Store the neighbor with its base cost
                network[node].append((new_row, new_col, base_cost))

print(f"Network created with {len(network)} nodes")
print(f"Start node {START_POINT} has {len(network[START_POINT])} neighbors")
print(f"End node {END_POINT} has {len(network[END_POINT])} neighbors")

# --- Check if the end point is reachable from the start ---
def is_reachable(start, end, grid, directions):
    """Simple BFS to check if the end point is reachable from the start"""
    visited = set()
    queue = [start]
    
    while queue:
        current = queue.pop(0)
        if current == end:
            return True
        if current in visited:
            continue
        visited.add(current)
        
        # Add neighbors
        r, c = current
        for dr, dc in directions:
            new_r, new_c = r + dr, c + dc
            if (0 <= new_r < len(grid) and 
                0 <= new_c < len(grid[0]) and
                grid[new_r][new_c] == 0 and
                (new_r, new_c) not in visited):
                queue.append((new_r, new_c))
    
    return False

# Check reachability
if not is_reachable(START_POINT, END_POINT, grid, DIRECTIONS):
    print("ERROR: End point is not reachable from the start with the current grid!")
    # Create a path to make it reachable
    print("Creating a direct path from start to end...")
    direct_path = []
    r, c = START_POINT
    target_r, target_c = END_POINT
    while r != target_r or c != target_c:
        direct_path.append((r, c))
        # Move in the direction of the target
        move_r = 1 if r < target_r else (-1 if r > target_r else 0)
        move_c = 1 if c < target_c else (-1 if c > target_c else 0)
        r += move_r
        c += move_c
    direct_path.append(END_POINT)
    
    # Clear obstacles along this path
    for r, c in direct_path:
        grid[r, c] = 0
    print(f"Created path with {len(direct_path)} steps")
else:
    print("End point is reachable from the start with the current grid.")

# --- CP-SAT Model for Path Finding ---
model = cp_model.CpModel()

# Maximum path length needs to be large enough to allow reaching all required sites
# and the end point, but not too large to create inefficient solutions
# With the simple reachability test above, we know we need at least a certain path length
min_path_length = GRID_SIZE * 2  # Conservative estimate
MAX_PATH_LENGTH = min_path_length  # Start with a smaller value
# Planning horizon
MAX_TIME = MAX_PATH_LENGTH * 3  # Allow enough time for the path

# Now define time intervals for visualization based on MAX_TIME
time_intervals = []
for period in range(time_periods):
    start_time = (MAX_TIME * period) // time_periods
    end_time = (MAX_TIME * (period + 1)) // time_periods if period < time_periods - 1 else MAX_TIME
    time_intervals.append({
        "name": time_interval_info[period]["name"],
        "start": start_time,
        "end": end_time,
        "speed_factor": time_interval_info[period]["speed_factor"]
    })

# --- Variables ---
# Whether node (r,c) is visited at position p in the path
visit = {}
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        if grid[r, c] == 0:  # Only for non-obstacle cells
            for p in range(MAX_PATH_LENGTH):
                visit[(r, c, p)] = model.NewBoolVar(f'visit_{r}_{c}_{p}')

# Time period for each position in the path
time_period = {}
for p in range(MAX_PATH_LENGTH):
    time_period[p] = model.NewIntVar(0, time_periods-1, f'time_period_{p}')

# Arrival time at each position in the path
arrival_time = {}
for p in range(MAX_PATH_LENGTH):
    arrival_time[p] = model.NewIntVar(0, MAX_TIME, f'time_{p}')

# Whether the path ends at position p
path_ends = {}
for p in range(MAX_PATH_LENGTH):
    path_ends[p] = model.NewBoolVar(f'path_ends_{p}')

# Final path length
path_length = model.NewIntVar(1, MAX_PATH_LENGTH, 'path_length')

# Simplify the model to ensure a feasible solution
# --- Path Constraints ---
# Start point constraint - the path must start at START_POINT
model.Add(visit[(START_POINT[0], START_POINT[1], 0)] == 1)

# Time starts at 0
model.Add(arrival_time[0] == 0)

# Each position in the path has exactly one node (or none if path ends)
for p in range(MAX_PATH_LENGTH):
    # Create a list of all visit variables for position p
    position_vars = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r, c] == 0:  # Only for non-obstacle cells
                position_vars.append(visit[(r, c, p)])
    
    # If the path hasn't ended, exactly one node is visited
    model.Add(sum(position_vars) == 1).OnlyEnforceIf(path_ends[p].Not())
    # If the path has ended, no node is visited
    model.Add(sum(position_vars) == 0).OnlyEnforceIf(path_ends[p])

# Path continuity: Can only move to adjacent nodes
for p in range(MAX_PATH_LENGTH - 1):
    # For each cell (r, c) at position p
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r, c] == 1:  # Skip obstacles
                continue
            
            # If we're at (r, c) at position p, the next position must be a neighbor
            neighbors = network.get((r, c), [])
            neighbor_vars = []
            
            for next_r, next_c, _ in neighbors:
                neighbor_vars.append(visit[(next_r, next_c, p+1)])
            
            # If the path ends at p, there's no next position
            model.Add(sum(neighbor_vars) == 0).OnlyEnforceIf(path_ends[p])
            
            # If we're at (r, c) at position p and the path continues,
            # we must move to one of the neighbors at position p+1
            # Create a condition for being at position (r,c) at step p
            at_rc_p = model.NewBoolVar(f'at_{r}_{c}_{p}')
            model.Add(visit[(r, c, p)] == 1).OnlyEnforceIf(at_rc_p)
            model.Add(visit[(r, c, p)] == 0).OnlyEnforceIf(at_rc_p.Not())
            
            # Create a condition for the path not ending at p
            path_not_ended_p = model.NewBoolVar(f'path_not_ended_{p}')
            model.Add(path_ends[p] == 0).OnlyEnforceIf(path_not_ended_p)
            model.Add(path_ends[p] == 1).OnlyEnforceIf(path_not_ended_p.Not())
            
            # If at (r,c) at p and path not ended, must visit a neighbor
            at_rc_p_and_not_ended = model.NewBoolVar(f'at_{r}_{c}_{p}_and_not_ended')
            model.AddBoolAnd([at_rc_p, path_not_ended_p]).OnlyEnforceIf(at_rc_p_and_not_ended)
            model.AddBoolOr([at_rc_p.Not(), path_not_ended_p.Not()]).OnlyEnforceIf(at_rc_p_and_not_ended.Not())
            
            # When both conditions are true, ensure we go to exactly one neighbor
            if neighbor_vars:  # Only if there are neighbors
                model.Add(sum(neighbor_vars) == 1).OnlyEnforceIf(at_rc_p_and_not_ended)

# SIMPLIFIED APPROACH: We'll add these constraints:
# 1. END_POINT must be visited exactly once before the path ends
end_point_visits = []
for p in range(MAX_PATH_LENGTH):
    if (END_POINT[0], END_POINT[1], p) in visit:
        end_point_visits.append(visit[(END_POINT[0], END_POINT[1], p)])
model.Add(sum(end_point_visits) == 1)

# 2. When we're at END_POINT, the path ends in the next step
for p in range(MAX_PATH_LENGTH - 1):
    at_end_point = model.NewBoolVar(f'at_end_point_{p}')
    model.Add(visit[(END_POINT[0], END_POINT[1], p)] == 1).OnlyEnforceIf(at_end_point)
    model.Add(visit[(END_POINT[0], END_POINT[1], p)] == 0).OnlyEnforceIf(at_end_point.Not())
    
    # If at END_POINT at step p, then path_ends[p+1] must be 1
    model.Add(path_ends[p+1] == 1).OnlyEnforceIf(at_end_point)

# 3. Path remains ended once it ends
for p in range(MAX_PATH_LENGTH - 1):
    model.AddImplication(path_ends[p], path_ends[p+1])

# All required sites must be visited (if MUST_VISIT_SITES is True)
if MUST_VISIT_SITES:
    print(f"Adding constraints to visit {len(REQUIRED_SITES)} required sites")
    
    # For each required site, it must be visited at some position in the path
    for site in REQUIRED_SITES:
        site_r, site_c = site
        
        # Add a stronger constraint to ensure required sites are visited
        # Create variables indicating if site is visited at position p
        site_visit_pos = []
        
        # Make sure the site is valid (not an obstacle)
        if grid[site_r, site_c] == 0:
            for p in range(MAX_PATH_LENGTH):
                if (site_r, site_c, p) in visit:
                    site_visit_pos.append(visit[(site_r, site_c, p)])
            
            # Site must be visited at exactly one position
            if site_visit_pos:
                model.Add(sum(site_visit_pos) == 1)
                print(f"Added constraint to visit required site at ({site_r}, {site_c})")
            else:
                print(f"Warning: Required site at ({site_r}, {site_c}) cannot be visited (invalid coordinates)")
        else:
            print(f"Error: Required site at ({site_r}, {site_c}) is an obstacle - cannot be visited")
            # Fix the grid - required sites cannot be obstacles
            grid[site_r, site_c] = 0

# Path ends once and remains ended
for p in range(MAX_PATH_LENGTH - 1):
    # If the path has ended at position p, it remains ended at position p+1
    model.AddImplication(path_ends[p], path_ends[p+1])

# Determine path length
for p in range(MAX_PATH_LENGTH):
    # Create a new bool var for the condition
    is_end_point = model.NewBoolVar(f'is_end_point_{p}')
    
    if p == 0:
        # If p==0, we only need to check if path_ends[0] is true
        model.AddBoolAnd([path_ends[p]]).OnlyEnforceIf(is_end_point)
        model.AddBoolOr([path_ends[p].Not()]).OnlyEnforceIf(is_end_point.Not())
    else:
        # If p>0, we need to check if path_ends[p] is true AND path_ends[p-1] is false
        # Create intermediate variables
        prev_not_ended = model.NewBoolVar(f'prev_not_ended_{p}')
        model.Add(path_ends[p-1] == 0).OnlyEnforceIf(prev_not_ended)
        model.Add(path_ends[p-1] == 1).OnlyEnforceIf(prev_not_ended.Not())
        
        # Now combine with current path_ends
        model.AddBoolAnd([path_ends[p], prev_not_ended]).OnlyEnforceIf(is_end_point)
        model.AddBoolOr([path_ends[p].Not(), prev_not_ended.Not()]).OnlyEnforceIf(is_end_point.Not())
    
    # Set path_length if this is the end point
    model.Add(path_length == p + 1).OnlyEnforceIf(is_end_point)

# Simplified time propagation model
# We'll just use a fixed cost for each step based on the time period,
# rather than a complex time-dependent model

# For each position in the path
for p in range(MAX_PATH_LENGTH - 1):
    # If the path hasn't ended at position p
    not_ended_at_p = model.NewBoolVar(f'not_ended_at_p_{p}')
    model.Add(path_ends[p] == 0).OnlyEnforceIf(not_ended_at_p)
    model.Add(path_ends[p] == 1).OnlyEnforceIf(not_ended_at_p.Not())
    
    # When not ended, add a constant cost for the time period
    for t in range(time_periods):
        in_period_t = model.NewBoolVar(f'in_period_t_{p}_{t}')
        model.Add(time_period[p] == t).OnlyEnforceIf(in_period_t)
        
        # If not ended and in period t, add cost based on period
        not_ended_in_period_t = model.NewBoolVar(f'not_ended_in_period_t_{p}_{t}')
        model.AddBoolAnd([not_ended_at_p, in_period_t]).OnlyEnforceIf(not_ended_in_period_t)
        model.AddBoolOr([not_ended_at_p.Not(), in_period_t.Not()]).OnlyEnforceIf(not_ended_in_period_t.Not())
        
        # Add a base cost for this period - use a simple constant per period
        # Period 0: Cost 10, Period 1: Cost 20, Period 2: Cost 30
        period_cost = (t + 1) * 10
        model.Add(arrival_time[p+1] == arrival_time[p] + period_cost).OnlyEnforceIf(not_ended_in_period_t)
    
    # If the path has ended, time doesn't advance
    model.Add(arrival_time[p+1] == arrival_time[p]).OnlyEnforceIf(not_ended_at_p.Not())

# Simplified time period determination using absolute arrival times
# We'll just use fixed thresholds for each period
for p in range(MAX_PATH_LENGTH):
    # Period boundaries (simple fixed values)
    # Period 0: 0-100, Period 1: 101-200, Period 2: 201+
    period_thresholds = [0, 100, 200, MAX_TIME]
    
    # For each period, check if arrival_time is within its bounds
    for t in range(time_periods):
        period_t = model.NewBoolVar(f'in_period_{t}_pos_{p}')
        
        # Check if arrival_time is within the appropriate range
        lower_bound = period_thresholds[t]
        upper_bound = period_thresholds[t+1]
        
        # Add the appropriate constraints
        if t < time_periods - 1:  # For all periods except the last
            # arrival_time must be >= lower_bound and < upper_bound
            model.Add(arrival_time[p] >= lower_bound).OnlyEnforceIf(period_t)
            model.Add(arrival_time[p] < upper_bound).OnlyEnforceIf(period_t)
        else:  # For the last period
            # arrival_time must be >= lower_bound
            model.Add(arrival_time[p] >= lower_bound).OnlyEnforceIf(period_t)
        
        # Set time_period[p] to t when in period t
        model.Add(time_period[p] == t).OnlyEnforceIf(period_t)

# Create a variable to indicate if we reach the end point
reaches_end = model.NewBoolVar('reaches_end')

# Add a large reward for reaching the end point
# Since we want to minimize the objective, we subtract the reward
# Create a very large penalty (negative reward) for not reaching the end
end_point_reward = 1000000

# This is a boolean variable that is true when we need to apply the penalty
apply_penalty = model.NewBoolVar('apply_penalty')

# Condition to check if the path reaches the end
# Check if there exists a position p where path_ends[p] is true
end_point_reached = []
for p in range(MAX_PATH_LENGTH):
    # Add this position's path_ends to the list
    end_point_reached.append(path_ends[p])

# If any path_ends[p] is true, then reaches_end should be true
model.AddBoolOr(end_point_reached).OnlyEnforceIf(reaches_end)
model.AddBoolAnd([var.Not() for var in end_point_reached]).OnlyEnforceIf(reaches_end.Not())

# If reaches_end is false, apply the penalty
model.AddBoolAnd([reaches_end.Not()]).OnlyEnforceIf(apply_penalty)
model.AddBoolAnd([reaches_end]).OnlyEnforceIf(apply_penalty.Not())

# --- Determine path length ---
# Set path_length based on when the path ends
for p in range(MAX_PATH_LENGTH):
    is_path_end = model.NewBoolVar(f'is_path_end_{p}')
    
    # This is the path end if path_ends[p] is true and 
    # either p=0 or path_ends[p-1] is false
    if p == 0:
        # If p=0, just check if path_ends[0] is true
        model.Add(path_ends[p] == 1).OnlyEnforceIf(is_path_end)
        model.Add(path_ends[p] == 0).OnlyEnforceIf(is_path_end.Not())
    else:
        # Create intermediate variable for path_ends[p-1] == 0
        prev_not_ended = model.NewBoolVar(f'prev_not_ended_{p}')
        model.Add(path_ends[p-1] == 0).OnlyEnforceIf(prev_not_ended)
        model.Add(path_ends[p-1] == 1).OnlyEnforceIf(prev_not_ended.Not())
        
        # is_path_end is true if path_ends[p] is true and path_ends[p-1] is false
        model.AddBoolAnd([path_ends[p], prev_not_ended]).OnlyEnforceIf(is_path_end)
        model.AddBoolOr([path_ends[p].Not(), prev_not_ended.Not()]).OnlyEnforceIf(is_path_end.Not())
    
    # Set path_length if this is where the path ends
    model.Add(path_length == p + 1).OnlyEnforceIf(is_path_end)

# --- Objective: Minimize Path Length ---
# Simple objective: just minimize the path length
# This ensures we get the shortest path that visits all required sites
model.Minimize(path_length)

# --- Solve the Model ---
print("Solving the model...")
start_time = time.time()
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60  # Set a time limit
status = solver.Solve(model)
end_time = time.time()
print(f"Solve time: {end_time - start_time:.2f} seconds")

# --- Process and Visualize the Solution ---
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"Solution found with status: {solver.StatusName(status)}")
    print(f"Objective value (total travel time): {solver.ObjectiveValue()}")
    print(f"Path length: {solver.Value(path_length)}")
    
    # Extract the path
    final_path = []
    for p in range(MAX_PATH_LENGTH):
        found = False
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r, c] == 0 and (r, c, p) in visit and solver.Value(visit[(r, c, p)]) == 1:
                    final_path.append((r, c))
                    found = True
                    break
            if found:
                break
        
        # Check if path ends
        if p < MAX_PATH_LENGTH and solver.Value(path_ends[p]) == 1:
            # Make sure we include the END_POINT if it's not already added
            if final_path[-1] != END_POINT:
                final_path.append(END_POINT)
            break
            
    # Make sure the path includes the end point
    if final_path[-1] != END_POINT:
        print("Warning: Path doesn't reach the end point, adding it manually")
        final_path.append(END_POINT)
    
    # Print the path with timing information
    print("\nPath details:")
    for p, (r, c) in enumerate(final_path):
        # For manually added points (after the solver's path), use the last valid period and time
        if p < len(final_path) - 1 or p < MAX_PATH_LENGTH and p in time_period and p in arrival_time:
            try:
                period = solver.Value(time_period[p])
                time_at_p = solver.Value(arrival_time[p])
            except (KeyError, ValueError):
                # If we added the end point manually, use the last known values
                if p > 0:
                    period = solver.Value(time_period[p-1])
                    time_at_p = solver.Value(arrival_time[p-1])
                else:
                    period = 0
                    time_at_p = 0
        else:
            # For manually added end points
            period = solver.Value(time_period[MAX_PATH_LENGTH-1]) if MAX_PATH_LENGTH-1 in time_period else 0
            time_at_p = solver.Value(arrival_time[MAX_PATH_LENGTH-1]) if MAX_PATH_LENGTH-1 in arrival_time else 0
            
        print(f"Step {p}: Cell ({r}, {c}), Time period: {period}, Arrival time: {time_at_p}")
    
    # --- Visualization ---
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 1. Visualize the grid with obstacles and path
    obstacle_grid = np.copy(grid)
    path_grid = np.zeros_like(grid)
    
    for p, (r, c) in enumerate(final_path):
        path_grid[r, c] = p + 1  # Mark the path with step numbers
    
    # Custom color map for path
    colors = ['white', 'blue', 'green', 'red']
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
    
    # Plot the obstacle grid with clear black obstacles
    obstacle_grid_display = np.copy(obstacle_grid)
    # Convert 1s to higher values for better contrast
    obstacle_grid_display[obstacle_grid_display == 1] = 10
    
    obstacle_cmap = LinearSegmentedColormap.from_list('obstacles', ['white', 'black'], N=11)
    ax1.imshow(obstacle_grid_display, cmap=obstacle_cmap, alpha=0.7)
    
    # Plot the path grid
    path_plot = ax1.imshow(path_grid, cmap=cmap, alpha=0.7)
    plt.colorbar(path_plot, ax=ax1, label='Path Step')
    
    # Add start and end markers
    ax1.plot(START_POINT[1], START_POINT[0], 'go', markersize=10, label='Start')
    ax1.plot(END_POINT[1], END_POINT[0], 'ro', markersize=10, label='End')
    
    # Add required sites markers
    for r, c in REQUIRED_SITES:
        ax1.plot(c, r, color='orange', markersize=10, marker='*', label=f'Required Site ({r},{c})')
    
    # Add grid lines
    ax1.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax1.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    # Configure the plot
    ax1.set_title('Grid with Path')
    ax1.legend()
    
    # 2. Visualize the path as a line plot with time periods
    path_x = [c for r, c in final_path]
    path_y = [r for r, c in final_path]
    
    # Get time periods for coloring
    time_periods_list = []
    for p in range(len(final_path)):
        # For manually added points (after the solver's path), use the last valid period
        try:
            if p < MAX_PATH_LENGTH:
                period = solver.Value(time_period[p])
            else:
                # Use the last known period for manually added points
                period = time_periods_list[-1] if time_periods_list else 0
        except (KeyError, ValueError):
            # If we added the end point manually, use the last known values
            period = time_periods_list[-1] if time_periods_list else 0
            
        time_periods_list.append(period)
    
    # Plot the path segments with colors based on time periods
    for i in range(len(path_x) - 1):
        period = time_periods_list[i]
        if period == 0:
            color = 'blue'
        elif period == 1:
            color = 'green'
        else:
            color = 'red'
        
        ax2.plot([path_x[i], path_x[i+1]], [path_y[i], path_y[i+1]], 
                 color=color, linewidth=2, alpha=0.7)
    
    # Add markers for each step with step numbers
    for i, (x, y) in enumerate(zip(path_x, path_y)):
        ax2.plot(x, y, 'o', markersize=8, alpha=0.7)
        ax2.text(x, y, str(i), fontsize=9, ha='center', va='center')
    
    # Add start and end markers
    ax2.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
    ax2.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='End')
    
    # Add required sites
    for r, c in REQUIRED_SITES:
        ax2.plot(c, r, color='orange', markersize=12, marker='*', label=f'Required Site ({r},{c})')
    
    # Configure the second plot
    ax2.set_title('Path with Time Periods')
    ax2.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax2.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax2.invert_yaxis()  # Invert y-axis to match grid coordinates
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.grid(True)
    
    # Add a custom legend for time periods
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', label='Period 0'),
        Patch(facecolor='green', edgecolor='green', label='Period 1'),
        Patch(facecolor='red', edgecolor='red', label='Period 2')
    ]
    # Place legend at the top of the plot to avoid blocking the path
    ax2.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    plt.tight_layout()
    plt.savefig('grid_network_path.png', dpi=300)
    plt.show()
    
    # --- 3D Visualization of Time-Dependent Path ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract time values for the z-coordinate
    times = []
    for p in range(len(final_path)):
        # Handle cases where we might have added points manually
        try:
            if p < MAX_PATH_LENGTH:
                time_val = solver.Value(arrival_time[p])
            else:
                # Use the last known time for manually added points
                time_val = times[-1] if times else 0
        except (KeyError, ValueError):
            # If we added the end point manually, use the last known values
            time_val = times[-1] if times else 0
            
        times.append(time_val)
    
    # Create segment-colored 3D path based on time periods
    for i in range(len(final_path) - 1):
        # Get period for this segment (use the precomputed period)
        period = time_periods_list[i]
        color = ['blue', 'green', 'red'][period]
        
        # Plot segment with appropriate color
        ax.plot([path_x[i], path_x[i+1]], 
                [path_y[i], path_y[i+1]], 
                [times[i], times[i+1]], 
                color=color, linewidth=3, alpha=0.8)
    
    # Add markers for each step
    for i, (x, y, z) in enumerate(zip(path_x, path_y, times)):
        node_type = 'regular'
        if i == 0:
            node_type = 'start'
        elif i == len(path_x) - 1:
            node_type = 'end'
        elif (final_path[i][0], final_path[i][1]) in REQUIRED_SITES:
            node_type = 'required'
        
        # Color by node type
        if node_type == 'start':
            color = 'green'
            marker = 'o'
            size = 100
        elif node_type == 'end':
            color = 'red'
            marker = 'o'
            size = 100
        elif node_type == 'required':
            color = 'orange'
            marker = '*'
            size = 150
        else:
            color = 'blue'
            marker = 'o'
            size = 50
            
        # Plot the point with appropriate styling
        ax.scatter(x, y, z, color=color, s=size, marker=marker, zorder=10)
        
        # Add step numbers (labels)
        if node_type == 'start' or node_type == 'end' or node_type == 'required':
            # Make labels for important points more prominent
            ax.text(x, y, z+2, f'{i}', color='black', fontsize=10, weight='bold')
        else:
            ax.text(x, y, z+1, f'{i}', color='black', fontsize=8)
    
    # Add the grid plane at z=0
    x_grid, y_grid = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE))
    z_grid = np.zeros_like(x_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.15, color='gray')
    
    # Add time interval planes with labels
    for interval in time_intervals:
        # Plot a semi-transparent plane at each time interval boundary
        interval_z = interval["start"]
        if interval_z > 0:  # Skip the first interval which starts at 0
            xx, yy = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE))
            zz = np.ones_like(xx) * interval_z
            ax.plot_surface(xx, yy, zz, alpha=0.1, color='red')
            
            # Add interval label
            ax.text(GRID_SIZE//2, GRID_SIZE//2, interval_z, 
                   f'{interval["name"]}', color='red', fontsize=8,
                   horizontalalignment='center')
    
    # Mark obstacles clearly in black
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r, c] == 1:
                # Add more prominent markers for obstacles
                ax.scatter(c, r, 0, color='black', s=150, marker='x', linewidth=2, zorder=5)
                # Add a small black cube to make obstacles more visible
                cube_size = 0.4
                xx, yy = np.meshgrid([c-cube_size/2, c+cube_size/2], [r-cube_size/2, r+cube_size/2])
                zz = np.zeros_like(xx)
                ax.plot_surface(xx, yy, zz, color='black', alpha=0.6)
    
    # Add required sites that aren't in the path (if any)
    for r, c in REQUIRED_SITES:
        for i, (path_r, path_c) in enumerate(final_path):
            if (path_r, path_c) == (r, c):
                break
        else:
            # If not in path, mark it on the ground plane
            ax.scatter(c, r, 0, color='orange', s=150, marker='*')
    
    # Configure the 3D plot
    ax.set_title('3D Time-Dependent Path')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_zlabel('Time')
    ax.view_init(30, 45)  # Adjust viewing angle
    
    # Set axis limits to ensure the grid is visible
    ax.set_xlim([-0.5, GRID_SIZE - 0.5])
    ax.set_ylim([-0.5, GRID_SIZE - 0.5])
    
    # Add vertical gridlines for better spatial reference
    for i in range(GRID_SIZE):
        ax.plot([i, i], [0, GRID_SIZE-1], [0, 0], 'k-', alpha=0.2)
        ax.plot([0, GRID_SIZE-1], [i, i], [0, 0], 'k-', alpha=0.2)
    
    # Add a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='End'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='orange', markersize=12, label='Required Site'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=10, label='Obstacle'),
    ]
    
    # Add period colors to legend
    period_colors = ['blue', 'green', 'red']
    for period, color in enumerate(period_colors):
        if period < time_periods:
            legend_elements.append(Line2D([0], [0], color=color, lw=3, label=f'Period {period}'))
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('grid_network_path_3d.png', dpi=300)
    
    # Add a new gantt chart view of the schedule
    plt.figure(figsize=(14, 6))
    
    # Define colors for different time periods
    period_colors = ['blue', 'green', 'red']
    
    # Track node types for the legend
    node_types = {
        'Start': False,
        'End': False,
        'Required': False,
        'Regular': False
    }
    
    # Create bars for each segment
    for i in range(len(final_path) - 1):
        from_node = final_path[i]
        to_node = final_path[i+1]
        
        # Safely get start_time
        try:
            if i < MAX_PATH_LENGTH:
                start_time = solver.Value(arrival_time[i])
            else:
                start_time = 0
        except (KeyError, ValueError):
            start_time = 0
            
        # Safely get end_time
        try:
            if i+1 < MAX_PATH_LENGTH:
                end_time = solver.Value(arrival_time[i+1])
            else:
                # Use a slightly higher value than start_time
                end_time = start_time + 5
        except (KeyError, ValueError):
            # For manually added end points, just add a small increment
            end_time = start_time + 5
            
        # Safely get period
        period = time_periods_list[i]  # Use our precomputed list
        
        # Plot movement segment
        plt.barh(0, end_time - start_time, left=start_time, height=0.5, 
                color=period_colors[period], alpha=0.7,
                edgecolor='black', linewidth=1)
        
        # Add node labels
        plt.text((start_time + end_time) / 2, 0, 
                f"{i}->{i+1}\n{from_node}->{to_node}", 
                ha='center', va='center', fontsize=8)
        
        # Add markers for different node types
        if i == 0:
            plt.scatter(start_time, 0, color='green', s=100, marker='o', zorder=10)
            node_types['Start'] = True
        elif i == len(final_path) - 2:
            plt.scatter(end_time, 0, color='red', s=100, marker='o', zorder=10)
            node_types['End'] = True
        
        # Mark required sites
        if from_node in REQUIRED_SITES:
            plt.scatter(start_time, 0, color='orange', s=100, marker='*', zorder=10)
            node_types['Required'] = True
        else:
            # Mark regular nodes with small dots
            if i > 0:  # Skip start point which is already marked
                plt.scatter(start_time, 0, color='blue', s=30, marker='o', alpha=0.7, zorder=5)
                node_types['Regular'] = True
    
    # Add vertical lines for time intervals
    for interval in time_intervals:
        plt.axvline(x=interval["start"], color='gray', linestyle='--', alpha=0.5)
        plt.text(interval["start"], 0.7, 
                f"{interval['name']} (Factor: {interval['speed_factor']})", 
                rotation=90, va='bottom', ha='right', fontsize=8)
    
    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # Add node type elements to legend if they appear in the path
    if node_types['Start']:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start'))
    if node_types['End']:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='End'))
    if node_types['Required']:
        legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='orange', markersize=10, label='Required Site'))
    if node_types['Regular']:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, label='Regular Point'))
    
    # Add time period colors to legend
    for period, color in enumerate(period_colors):
        if period < time_periods:
            legend_elements.append(Line2D([0], [0], color=color, lw=4, label=f'Period {period}'))
    
    # Configure the plot
    plt.yticks([])  # Hide y-axis ticks since we only have one row
    plt.xlabel('Time')
    plt.title('Path Schedule with Time Periods')
    plt.grid(axis='x', alpha=0.3)
    
    # Adjust figure size to allow space for the legend at bottom
    plt.subplots_adjust(bottom=0.25)
    
    # Add the legend below the plot with enough space
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
    
    plt.savefig('grid_network_path_schedule.png', dpi=300)
    
    print("\nAll visualizations have been saved as PNG files.")
    
    # Only show the plots if you want to view them interactively
    # Comment out this line or set INTERACTIVE_PLOTTING to False 
    # when running in a non-interactive environment
    INTERACTIVE_PLOTTING = True
    if INTERACTIVE_PLOTTING:
        plt.show()
    else:
        plt.close('all')
    
else:
    print(f"No solution found. Status: {solver.StatusName(status)}")
    if status == cp_model.INFEASIBLE:
        print("The model is infeasible. Possible reasons:")
        print("1. There may be no valid path from start to end")
        print("2. The time constraints may be too restrictive")
        print("3. Check the model constraints for logical inconsistencies")