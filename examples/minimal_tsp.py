"""
Minimal Traveling Salesperson Problem (TSP) example that should be feasible.
This is a completely rewritten version to resolve the infeasibility issue.
"""

import math
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model

# --- 1. Problem Definition & Data ---
print("\n=== Minimal Traveling Salesperson Problem (TSP) Example ===")

# Locations (latitude, longitude)
locations = [
    (40.71, -74.00),  # Location 0 (NYC) - Used as the depot
    (40.75, -73.98),  # Location 1 (Times Square)
]

num_locations = len(locations)
print(f"Number of locations: {num_locations}")

# Calculate distances
dist_matrix = []
for i in range(num_locations):
    dist_row = []
    for j in range(num_locations):
        if i == j:
            dist_row.append(0)
        else:
            lat1, lon1 = locations[i]
            lat2, lon2 = locations[j]
            # Simple distance calculation
            distance = math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 5000
            dist_row.append(int(distance))
    dist_matrix.append(dist_row)

print("Distance matrix:")
for row in dist_matrix:
    print(row)

# --- 2. Model Creation ---
model = cp_model.CpModel()

# --- Variables ---
# x[i][j] = 1 if we travel from location i to location j
x = {}
for i in range(num_locations):
    for j in range(num_locations):
        if i != j:  # No self-loops
            x[i, j] = model.NewBoolVar(f'x_{i}_{j}')

# Time variables
# time[i] = arrival time at location i
time = {}
for i in range(num_locations):
    time[i] = model.NewIntVar(0, 10000, f'time_{i}')

# --- Constraints ---
# Each location must be visited exactly once
for j in range(num_locations):
    if j != 0:  # Skip for the depot
        model.Add(sum(x[i, j] for i in range(num_locations) if i != j) == 1)

# Each location must be left exactly once
for i in range(num_locations):
    if i != 0:  # Skip for the depot
        model.Add(sum(x[i, j] for j in range(num_locations) if j != i) == 1)

# Depot is the start and end
# Leave depot exactly once
model.Add(sum(x[0, j] for j in range(1, num_locations)) == 1)
# Return to depot exactly once
model.Add(sum(x[i, 0] for i in range(1, num_locations)) == 1)

# Time constraints - ensure path is connected
for i in range(num_locations):
    for j in range(1, num_locations):  # j=0 is the depot
        if i != j:
            # time[j] >= time[i] + distance[i][j] if x[i][j] = 1
            model.Add(time[j] >= time[i] + dist_matrix[i][j]).OnlyEnforceIf(x[i, j])

# Start at time 0 at depot
model.Add(time[0] == 0)

# --- Objective ---
# Minimize total distance
total_distance = []
for i in range(num_locations):
    for j in range(num_locations):
        if i != j:
            total_distance.append(model.NewIntVar(0, 10000, f'dist_{i}_{j}'))
            # dist_i_j = distance[i][j] if x[i][j] = 1, else 0
            model.Add(total_distance[-1] == dist_matrix[i][j]).OnlyEnforceIf(x[i, j])
            model.Add(total_distance[-1] == 0).OnlyEnforceIf(x[i, j].Not())

model.Minimize(sum(total_distance))

# --- 3. Solve ---
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 10.0
status = solver.Solve(model)

# --- 4. Output Solution ---
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("\nSolution found!")
    print(f"Status: {solver.StatusName(status)}")
    print(f"Objective value: {solver.ObjectiveValue()}")
    
    # Construct the tour
    tour = [0]  # Start at the depot
    current = 0
    while len(tour) < num_locations + 1:  # +1 because we return to depot
        for j in range(num_locations):
            if current != j and (current, j) in x and solver.Value(x[current, j]) == 1:
                tour.append(j)
                current = j
                break
    
    print(f"Tour: {tour}")
    print("Location times:")
    for i in range(num_locations):
        print(f"  Location {i}: {solver.Value(time[i])}")
    
    # Plot the solution
    plt.figure(figsize=(10, 6))
    
    # Plot locations
    for i, (lat, lon) in enumerate(locations):
        if i == 0:  # Depot
            plt.scatter(lon, lat, c='red', s=100, marker='s', label='Depot')
        else:  # Other locations
            plt.scatter(lon, lat, c='blue', s=100, marker='o', label=f'Location {i}')
        plt.text(lon, lat, str(i), fontsize=12)
    
    # Plot tour
    for i in range(len(tour) - 1):
        from_idx = tour[i]
        to_idx = tour[i + 1]
        plt.plot([locations[from_idx][1], locations[to_idx][1]], 
                [locations[from_idx][0], locations[to_idx][0]], 
                'g-', linewidth=2)
    
    plt.title(f"TSP Solution - Total Distance: {solver.ObjectiveValue()}")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print(f"No solution found. Status: {solver.StatusName(status)}")
    
    if status == cp_model.INFEASIBLE:
        print("\nDumping model for debugging:")
        print(model.Proto())