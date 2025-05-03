import math
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model
import sys

# --- 1. Problem Definition & Data ---

# Locations (latitude, longitude - simplified for distance calc)
# Depots first, then sites
locations = [
    (40.71, -74.00),  # Depot 0 (NYC)
    (34.05, -118.24), # Depot 1 (LA) - Far away to force usage of Depot 0
    (40.75, -73.98),  # Site 2 (Times Square)
    (40.76, -73.97),  # Site 3 (Near TS)
    (40.68, -73.99),  # Site 4 (Brooklyn Bridge)
    (40.70, -73.95),  # Site 5 (Williamsburg)
    (40.80, -73.96),  # Site 6 (Upper West Side)
]

num_locations = len(locations)
num_vehicles = 2
num_depots = 2 # Indices 0 and 1 are depots
num_sites = num_locations - num_depots # Indices 2 to num_locations-1 are sites

# Vehicle Capacities (using maximum travel time as capacity)
# ***** DEBUGGING: Increase capacity significantly *****
# The previous value of 250 was likely too low given the calculated distances/times.
# Let's set it much higher to see if feasibility is achievable.
vehicle_time_capacity = 1500 # Increased substantially from 250

# Service time at each site (time units spent at the location)
service_times = [0] * num_depots + [10] * num_sites # No service time at depots

# Time discretization for varying costs
# Divide planning horizon into intervals (e.g., morning, afternoon, evening)
time_intervals = [
    {"name": "Morning", "start": 0, "end": 100, "speed_factor": 0.7}, # Slower
    {"name": "Afternoon", "start": 100, "end": 200, "speed_factor": 1.0}, # Normal
    # Increase end time of Evening interval to ensure it covers the increased capacity
    {"name": "Evening", "start": 200, "end": vehicle_time_capacity + 500, "speed_factor": 1.2}, # Faster, extend end time
]
num_intervals = len(time_intervals)
# Adjust planning horizon to match the extended interval and capacity
planning_horizon = time_intervals[-1]["end"] # Max time considered

print(f"--- Configuration ---")
print(f"Num Vehicles: {num_vehicles}")
print(f"Num Depots: {num_depots}")
print(f"Num Sites: {num_sites}")
print(f"Vehicle Time Capacity: {vehicle_time_capacity}")
print(f"Planning Horizon: {planning_horizon}")
print(f"Time Intervals: {time_intervals}")
print(f"--------------------\n")


# --- 2. Helper Functions ---

# Simplified Euclidean distance (scale factor to make times reasonable)
def calculate_distance(loc_idx1, loc_idx2):
    lat1, lon1 = locations[loc_idx1]
    lat2, lon2 = locations[loc_idx2]
    # Using a large scale factor for demonstration to get meaningful travel times
    return int(math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 5000)

# Calculate travel time based on distance AND time interval
# This is the core of time-dependency
travel_times = {} # Dictionary: (loc1, loc2, interval_idx) -> time
print("--- Precomputing Travel Times ---")
for i in range(num_locations):
    for j in range(num_locations):
        if i == j:
            for k in range(num_intervals):
                travel_times[(i, j, k)] = 0
            continue
        base_dist = calculate_distance(i, j)
        # print(f"Base distance {i} -> {j}: {base_dist}")
        for k in range(num_intervals):
            speed_factor = time_intervals[k]["speed_factor"]
            # Time = Distance / SpeedFactor (higher factor means faster/less time)
            # Ensure minimum travel time of 1 if not zero
            time = max(1, int(base_dist / speed_factor))
            travel_times[(i, j, k)] = time
            # print(f"  Interval {k} (Factor {speed_factor:.1f}): Time = {time}")
print("---------------------------------\n")

# Function to get interval index from a time value
def get_interval_index(time_value):
    # Clamp time_value to the max index of horizon_intervals if needed
    # Use max(0, ..) to handle potential negative intermediate values if model allows them temporarily
    clamped_time = min(int(max(0, time_value)), planning_horizon)

    for idx, interval in enumerate(time_intervals):
        # Check if time is within the interval [start, end)
        if interval["start"] <= clamped_time < interval["end"]:
            return idx
    # If time_value equals or exceeds the start of the last interval
    return num_intervals - 1

# Precompute interval indices for the horizon for use in AddElement
# Extend range to include planning_horizon
# Ensure planning_horizon is int for range()
horizon_intervals = [get_interval_index(t) for t in range(int(planning_horizon) + 1)]

# --- 3. Model Creation (CP-SAT) ---

model = cp_model.CpModel()

# --- Variables ---

# x[i, j, v]: vehicle 'v' travels from location 'i' to location 'j'
x = {}
for i in range(num_locations):
    for j in range(num_locations):
        # No travel from a node back to itself needed for standard VRP
        if i == j: continue
        for v in range(num_vehicles):
            x[i, j, v] = model.NewBoolVar(f'x_{i}_{j}_{v}')

# t[loc, v]: Arrival time of vehicle 'v' at location 'loc'
t = {}
for i in range(num_locations):
    for v in range(num_vehicles):
        # Start time at depots can be 0, arrival at sites > 0
        # Ensure domain covers the full horizon
        t[i, v] = model.NewIntVar(0, int(planning_horizon), f't_{i}_{v}')

# d[loc, v]: Departure time of vehicle 'v' from location 'loc'
d = {}
for i in range(num_locations):
     for v in range(num_vehicles):
        # Define as separate variable, then link with Add
        # Ensure domain covers the full horizon
        d[i, v] = model.NewIntVar(0, int(planning_horizon), f'd_{i}_{v}')
        # Departure = Arrival + Service Time (using Add for robust propagation)
        model.Add(d[i, v] == t[i, v] + service_times[i])

# --- Intermediate/Helper Variables ---
# Keep track of variables needed later
vehicle_serves_any_site = {} # Dict: v -> BoolVar
vehicle_ends = {} # Dict: v -> List[BoolVar] (indicators for ending at each depot)
vehicle_final_depot_arrival = {} # Dict: v -> IntVar

# --- Constraints ---

depot_indices = list(range(num_depots))
site_indices = list(range(num_depots, num_locations))

# 1. Routing Constraints (Assign sites to vehicles, ensure flow)
# Each site is visited exactly once by some vehicle
for j in site_indices:
    # Ensure the sum includes only valid arcs (i != j)
    model.Add(sum(x[i, j, v] for i in range(num_locations) if i != j for v in range(num_vehicles)) == 1)

# Flow conservation: For each vehicle and site, inflow equals outflow
for j in site_indices:
    for v in range(num_vehicles):
        inflow = sum(x[i, j, v] for i in range(num_locations) if i != j)
        outflow = sum(x[j, k, v] for k in range(num_locations) if k != j)
        model.Add(inflow == outflow)

# Each vehicle starts at *exactly one* depot and serves at least one site
vehicle_starts = {}
for v in range(num_vehicles):
    start_vars = []
    # Create list of arcs from depots to sites for this vehicle
    all_outgoing_from_depots_v = [x[dep,site_idx,v] for dep in depot_indices for site_idx in site_indices]

    for i in depot_indices:
        # Var indicating vehicle v starts at depot i (by leaving it for a site)
        starts_at_i = model.NewBoolVar(f'starts_at_{i}_v{v}')
        # Link starts_at_i to the existence of an arc x[i, j, v] where j is a site
        sites_served_from_i = [x[i, j, v] for j in site_indices]
        model.Add(sum(sites_served_from_i) >= 1).OnlyEnforceIf(starts_at_i) # Use >= 1
        model.Add(sum(sites_served_from_i) == 0).OnlyEnforceIf(starts_at_i.Not())
        start_vars.append(starts_at_i)

        # If vehicle v starts at depot i, its departure time (and arrival) must be 0
        model.Add(t[i, v] == 0).OnlyEnforceIf(starts_at_i) # Arrival T=0
        model.Add(d[i, v] == 0).OnlyEnforceIf(starts_at_i) # Departure T=0 (since service_time=0 at depots)

    # Determine if the vehicle serves *any* site
    serves_any_site_v = model.NewBoolVar(f'serves_any_site_v{v}') # Create the variable
    vehicle_serves_any_site[v] = serves_any_site_v # Store the variable
    model.Add(sum(all_outgoing_from_depots_v) >= 1).OnlyEnforceIf(serves_any_site_v) # Use >= 1
    model.Add(sum(all_outgoing_from_depots_v) == 0).OnlyEnforceIf(serves_any_site_v.Not())

    # Only enforce the single start depot constraint if the vehicle is used
    model.Add(sum(start_vars) == 1).OnlyEnforceIf(serves_any_site_v)
    # If vehicle is not used, ensure it doesn't start from any depot
    model.Add(sum(start_vars) == 0).OnlyEnforceIf(serves_any_site_v.Not())
    vehicle_starts[v] = start_vars


# Each vehicle ends at *exactly one* depot if it started
# vehicle_ends = {} # Moved definition up
for v in range(num_vehicles):
    end_vars = []
    # Create list of arcs from sites to depots for this vehicle
    all_incoming_to_depots_v = [x[site_idx, dep, v] for site_idx in site_indices for dep in depot_indices]

    for j in depot_indices:
        # Var indicating vehicle v ends at depot j (by arriving from a site)
        ends_at_j = model.NewBoolVar(f'ends_at_{j}_v{v}')
        sites_arriving_at_j = [x[i, j, v] for i in site_indices]
        model.Add(sum(sites_arriving_at_j) >= 1).OnlyEnforceIf(ends_at_j) # Use >= 1
        model.Add(sum(sites_arriving_at_j) == 0).OnlyEnforceIf(ends_at_j.Not())
        end_vars.append(ends_at_j)

    # If the vehicle serves any site, it must end at exactly one depot
    serves_any_site_v = vehicle_serves_any_site[v] # Retrieve the variable from storage
    model.Add(sum(end_vars) == 1).OnlyEnforceIf(serves_any_site_v)
    # If vehicle is not used, ensure it doesn't end at any depot
    model.Add(sum(end_vars) == 0).OnlyEnforceIf(serves_any_site_v.Not())
    vehicle_ends[v] = end_vars # Store end indicators


# 2. Time Propagation & Time-Dependent Cost Constraint
for i in range(num_locations):
    for j in range(num_locations):
        if i == j: continue
        for v in range(num_vehicles):
            # Get the departure time interval index from location i for vehicle v
            # Domain [0, num_intervals-1]
            dep_interval_idx = model.NewIntVar(0, num_intervals - 1, f'dep_interval_{i}_{j}_{v}') # Make name unique

            # Use AddElement: dep_interval_idx = horizon_intervals[d[i,v]]
            # Index var: d[i,v], Domain [0, planning_horizon]
            # List: horizon_intervals, Length planning_horizon+1
            # Target var: dep_interval_idx, Domain [0, num_intervals-1]
            model.AddElement(d[i, v], horizon_intervals, dep_interval_idx)

            # Get the correct travel time for that interval
            current_travel_time = model.NewIntVar(0, int(planning_horizon), f'travel_{i}_{j}_{v}')
            # Create a list of possible travel times from i to j for AddElement
            # List length is num_intervals, indexed by dep_interval_idx
            possible_times_ij = [travel_times[i,j,k] for k in range(num_intervals)]
            # Index var: dep_interval_idx, Domain [0, num_intervals-1]
            # List: possible_times_ij, Length num_intervals
            # Target var: current_travel_time, Domain [0, planning_horizon]
            model.AddElement(dep_interval_idx, possible_times_ij, current_travel_time)

            # If vehicle v travels from i to j (x[i,j,v] is true), then
            # arrival time at j must be departure time at i + travel time
            model.Add(t[j, v] == d[i, v] + current_travel_time).OnlyEnforceIf(x[i, j, v])


# 3. Vehicle Capacity Constraint (Max travel time)
# The total time elapsed when arriving back at the final depot <= capacity
for v in range(num_vehicles):
    # Ensure domain covers the full horizon + capacity
    final_depot_arrival_time_v = model.NewIntVar(0, int(vehicle_time_capacity), f'final_depot_arr_{v}')
    vehicle_final_depot_arrival[v] = final_depot_arrival_time_v # Store variable
    temp_arrival_times_at_depots = []

    # Check arrival time at each potential end depot j
    for j_idx, j in enumerate(depot_indices):
        # Get the indicator variable 'ends_at_j' created earlier
        ends_at_j = vehicle_ends[v][j_idx] # vehicle_ends was stored earlier

        # Create an intermediate variable to hold arrival time only if ends_at_j is true
        # Domain must match the target final_depot_arrival_time_v
        arrival_if_ends_here = model.NewIntVar(0, int(vehicle_time_capacity), f'arr_if_ends_{j}_v{v}')
        # If ends_at_j is true, arrival_if_ends_here takes the value of t[j, v]
        # Note: t[j,v] has domain [0, planning_horizon], ensure it's compatible
        model.Add(arrival_if_ends_here == t[j, v]).OnlyEnforceIf(ends_at_j)
        # If ends_at_j is false, arrival_if_ends_here is 0
        model.Add(arrival_if_ends_here == 0).OnlyEnforceIf(ends_at_j.Not())
        temp_arrival_times_at_depots.append(arrival_if_ends_here)

    # The final arrival time is the maximum of the arrivals at the specific depot where the vehicle ends.
    model.AddMaxEquality(final_depot_arrival_time_v, temp_arrival_times_at_depots)

    # Enforce the capacity constraint on the final arrival time variable itself (already done by its domain)
    # model.Add(final_depot_arrival_time_v <= vehicle_time_capacity) # This is redundant due to the var domain


# --- Objective Function ---
# Minimize makespan (maximum arrival time at any *final* depot for any *used* vehicle)
# Domain must cover the possible range of arrival times
makespan = model.NewIntVar(0, int(vehicle_time_capacity), 'makespan')

# Collect the final_depot_arrival_time variables calculated for the capacity constraint
all_final_depot_arrivals = [vehicle_final_depot_arrival[v] for v in range(num_vehicles)]

if all_final_depot_arrivals: # Ensure list is not empty
     model.AddMaxEquality(makespan, all_final_depot_arrivals)
     model.Minimize(makespan)
else:
     model.Minimize(model.NewConstant(0))


# --- 4. Solve ---
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60.0
solver.parameters.log_search_progress = True
# Optional: Increase number of workers if you have cores available
# solver.parameters.num_search_workers = 8
status = solver.Solve(model)

# --- 5. Output and Visualization ---

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"\nSolution Found (Status: {solver.StatusName(status)})")
    print(f"Objective (Makespan): {solver.ObjectiveValue()}")

    vehicle_routes = {v: [] for v in range(num_vehicles)}
    vehicle_times = {v: [] for v in range(num_vehicles)} # List of (Node, Arrival, Departure) tuples

    # Extract routes and times
    print("\n--- Routes and Schedules ---")
    for v in range(num_vehicles):
        # Check if vehicle v is used
        serves_any_site_v = vehicle_serves_any_site[v] # Retrieve variable
        if not solver.Value(serves_any_site_v):
            print(f"Vehicle {v}: Not used.")
            continue

        # Find the starting depot for vehicle v
        start_node = -1
        for i in depot_indices:
             # Check if vehicle v starts at depot i by checking outflow to sites
             if sum(solver.Value(x[i,j,v]) for j in site_indices if f'x_{i}_{j}_{v}' in x) > 0:
                 start_node = i
                 break

        if start_node == -1:
             print(f"Error: Vehicle {v} marked as used but start node not found.")
             continue # Should not happen if model is consistent

        print(f"Vehicle {v} starting at Depot {start_node}:")
        route = [start_node]
        times = [(start_node, solver.Value(t[start_node, v]), solver.Value(d[start_node, v]))]
        current_node = start_node

        visited_nodes = {start_node} # Keep track to prevent infinite loops in case of error
        max_steps = num_locations + 2 # Safety break for route tracing
        step_count = 0

        while step_count < max_steps:
            step_count += 1
            found_next = False
            # Find next node visited by vehicle v
            possible_next_nodes = site_indices + depot_indices # Can go to site or back to depot
            for next_node in possible_next_nodes:
                arc_var_name = f'x_{current_node}_{next_node}_{v}'
                # Ensure arc exists (i!=j) and is selected in the solution
                if current_node != next_node and arc_var_name in x and solver.Value(x[current_node, next_node, v]):

                    route.append(next_node)
                    visited_nodes.add(next_node) # Add to visited AFTER finding the arc

                    arr_time = solver.Value(t[next_node, v])
                    dep_time = solver.Value(d[next_node, v])

                    # Calculate and display the interval and travel time used for this leg
                    dep_time_prev = solver.Value(d[current_node, v])
                    interval_idx = get_interval_index(dep_time_prev)
                    travel_time_used = travel_times[(current_node, next_node, interval_idx)]
                    print(f"  {current_node} -> {next_node} | Dep@ {dep_time_prev:<5.1f} (Intvl {interval_idx}) | TravTime: {travel_time_used:<4} | Arr@ {arr_time:<5.1f} | Serv: {service_times[next_node]:<3} | Dep@ {dep_time:<5.1f}")

                    times.append((next_node, arr_time, dep_time))
                    current_node = next_node
                    found_next = True
                    break # Move to the next step in the route

            # Stop if we returned to a depot or if no next node is found
            if not found_next:
                if current_node not in depot_indices:
                    print(f"  Warning: Route for V{v} ended unexpectedly at node {current_node} (not a depot). Max steps: {step_count}.")
                break # Exit while loop

            if current_node in depot_indices:
                # Make sure this depot is the intended end depot based on indicator
                is_actual_end = False
                for j_idx, j in enumerate(depot_indices):
                     if j == current_node and solver.Value(vehicle_ends[v][j_idx]):
                         is_actual_end = True
                         break
                if is_actual_end:
                    print(f"  Route for V{v} ended at Depot {current_node}.")
                else:
                    # This might happen if a route goes Depot->Site->Depot (e.g., only one site visited)
                    # Check if it *started* at this depot too
                    if current_node == start_node and len(route) <= 2:
                         print(f"  Route for V{v} ended back at starting Depot {current_node} (short route).")
                    else:
                         print(f"  Warning: Route for V{v} arrived at Depot {current_node}, but it might not be the intended end depot? Continuing trace just in case (step {step_count}).")
                         # This case is tricky - should a vehicle pass *through* a depot?
                         # Current model doesn't explicitly allow/forbid depot passthrough.
                         # Assuming it ends when it hits *any* depot for now.
                         print(f"  Route for V{v} ended at Depot {current_node}.")

                break # Route finished when it reaches any depot

            if step_count >= max_steps:
                 print(f"  Error: Route tracing exceeded max steps ({max_steps}) for V{v}. Last node: {current_node}.")
                 break


        vehicle_routes[v] = route
        vehicle_times[v] = times

    # --- Plotting ---
    try: # Add try-except for plotting robustness
        plt.figure(figsize=(12, 10))

        # Plot locations
        depot_coords = [locations[i] for i in depot_indices]
        site_coords = [locations[i] for i in site_indices]
        plt.scatter([loc[1] for loc in depot_coords], [loc[0] for loc in depot_coords],
                    c='red', marker='s', s=150, label='Depots', zorder=5)
        plt.scatter([loc[1] for loc in site_coords], [loc[0] for loc in site_coords],
                    c='blue', marker='o', s=100, label='Sites', zorder=5)

        # Add labels to points
        for i, loc in enumerate(locations):
            plt.text(loc[1] + 0.002, loc[0] + 0.002, str(i), fontsize=12, zorder=6)

        # Plot routes
        colors = plt.cm.viridis([i / max(1, num_vehicles) for i in range(num_vehicles)]) # Avoid division by zero if num_vehicles=0
        for v in range(num_vehicles):
            route = vehicle_routes.get(v, []) # Use .get for safety
            times = vehicle_times.get(v, [])
            if not route or len(route) <= 1: continue # Skip unused vehicles

            route_coords = [locations[node_idx] for node_idx in route]
            route_lons = [coord[1] for coord in route_coords]
            route_lats = [coord[0] for coord in route_coords]

            # Add slight offset for overlapping routes if needed (optional)
            offset = v * 0.0005
            route_lons_offset = [lon + offset for lon in route_lons]
            route_lats_offset = [lat + offset for lat in route_lats]

            plt.plot(route_lons_offset, route_lats_offset,
                     color=colors[v], marker='>', markersize=5, linestyle='-', linewidth=2,
                     label=f'Vehicle {v}', zorder=3)

            # Annotate arrival times at sites
            for node_idx, arrival_time, departure_time in times:
                # Check if start/end depot based on the extracted route for this vehicle
                is_start_depot = (node_idx == route[0])
                is_end_depot = (node_idx == route[-1]) and node_idx in depot_indices

                # Annotate Sites
                if node_idx in site_indices:
                    plt.text(locations[node_idx][1] + 0.003 + offset, locations[node_idx][0] - 0.003 + offset,
                             f'Arr:{arrival_time:.0f}\nDep:{departure_time:.0f}',
                             fontsize=8, color=colors[v], ha='left', va='top', zorder=4,
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[v], alpha=0.8))
                # Annotate End Depot Arrival (if not also start depot)
                elif is_end_depot and not is_start_depot:
                     plt.text(locations[node_idx][1] + 0.003 + offset, locations[node_idx][0] - 0.003 + offset,
                             f'End Arr:\n{arrival_time:.0f}',
                             fontsize=8, color=colors[v], ha='left', va='top', zorder=4,
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[v], alpha=0.8))


        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        # Handle case where ObjectiveValue might not be available (e.g., infeasible)
        obj_value_str = f"{solver.ObjectiveValue():.1f}" if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else "N/A"
        plt.title(f"MDVRP with Time-Dependent Costs (Makespan: {obj_value_str}) - Status: {solver.StatusName(status)}")
        plt.legend(loc='best')
        plt.grid(True)
        # Adjust map bounds slightly for better visibility
        all_lons = [loc[1] for loc in locations]
        all_lats = [loc[0] for loc in locations]
        if all_lons and all_lats: # Ensure locations exist
            plt.xlim(min(all_lons) - 0.02, max(all_lons) + 0.02)
            plt.ylim(min(all_lats) - 0.02, max(all_lats) + 0.02)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"\nError during plotting: {e}")


elif status == cp_model.INFEASIBLE:
    print("Problem is Infeasible.")
    print("Possible reasons:")
    print(f" - Vehicle capacity ({vehicle_time_capacity}) might still be too low for distances/service times.")
    print(f" - Planning horizon ({planning_horizon}) might be too short (less likely now).")
    print(f" - Inconsistent constraints (e.g., start/end logic, time propagation).")
    # Dump model proto if infeasible to help debug
    # Be careful, this can be very large! Limit size or redirect to file if needed.
    try:
        print("\n--- Dumping Model Proto (potentially large) ---")
        print(model.Proto())
    except Exception as e:
        print(f"Error printing model proto: {e}")

elif status == cp_model.MODEL_INVALID:
    print("Model is Invalid. Check constraint definitions.")
else:
    print(f"Solver finished with status: {solver.StatusName(status)}")