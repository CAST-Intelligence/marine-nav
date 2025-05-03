**Product Requirements Document: USV Routing Optimization Tool**

**1. Introduction**

This document outlines the requirements for a Python-based optimization tool designed to generate efficient routes and schedules for fleets of Unmanned Surface Vehicles (USVs) performing survey tasks. The tool aims to minimize operational costs and/or total mission time (makespan) by intelligently assigning survey sites to USVs and determining their optimal paths, considering various constraints including time-dependent environmental factors (weather) affecting travel. The tool must address two primary operational scenarios: a classic multi-depot VRP and a more complex scenario involving Mobile Support Boats (MSBs).

**2. Goals and Objectives**

*   Develop a robust Python library/tool capable of solving two defined classes of USV routing problems.
*   Minimize total operational cost (e.g., fuel, time-based costs) or total mission duration (makespan), or a weighted combination.
*   Generate feasible and near-optimal route plans and schedules for all vehicles (USVs and MSBs).
*   Accurately model time-dependent travel costs/times based on provided forecast data.
*   Provide clear, machine-readable output detailing the planned routes, schedules, and key performance metrics.
*   Utilize Google OR-Tools where appropriate, particularly CP-SAT for its flexibility.
*   Address and resolve the known infeasibility bug in the current CP-SAT implementation for the classic MDVRP with time-dependent costs.

**3. User Stories**

*   As an Operations Planner, I want to input fleet characteristics (USVs, MSBs), fixed depot locations, survey site locations and durations, and time-varying weather forecasts, so that I can receive optimized deployment plans.
*   As an Operations Planner, I want to select between solving the classic MDVRP scenario (USVs operating from fixed depots) or the MSB-VRP scenario (USVs deployed/recovered by mobile boats).
*   As an Operations Planner, I want the tool to respect vehicle constraints like endurance (time/distance/energy), speed (including weather effects), and MSB carrying capacity.
*   As an Operations Planner, I want to define the optimization objective (minimize cost, minimize makespan, or a blend).
*   As an Operations Planner, I want the output to clearly show which vehicle visits which site, the sequence of visits, and the estimated arrival/departure times at each location, including rendezvous points for the MSB scenario.
*   As a Developer, I want a well-defined Python API to integrate this optimization logic into larger mission planning systems.

**4. Functional Requirements**

**4.1. Common Requirements**

*   **FR1: Input Handling:**
    *   Accept inputs via Python dictionaries/objects or potentially structured files (e.g., JSON, CSV).
    *   Required inputs:
        *   List of fixed depot locations (ID, coordinates).
        *   List of survey site locations (ID, coordinates, required survey time).
        *   List of USV definitions (ID, speed function/base speed, endurance limit [time/distance/energy], cost per unit time/distance, potentially specific payload affecting survey time).
        *   List of MSB definitions (if applicable) (ID, speed function/base speed, endurance limit, cost per unit time/distance, USV carrying capacity, USV deployment/recovery time penalty).
        *   Weather forecast data enabling time-dependent travel calculation (e.g., time intervals with associated speed modifiers or environmental parameters).
        *   Optimization objective selection (COST, TIME, WEIGHTED).
        *   Solver parameters (e.g., maximum solve time).
*   **FR2: Time-Dependent Travel Calculation:**
    *   Implement a core function `calculate_travel(origin_loc, dest_loc, departure_time, vehicle_type)` that returns travel duration and cost.
    *   This function *must* use the provided weather forecast data and vehicle speed characteristics to determine duration/cost based on the `departure_time`.
    *   Support discretization of the planning horizon based on forecast intervals.
*   **FR3: Output Generation:**
    *   Provide results in a structured format (e.g., Python dictionary/list of objects).
    *   Output must include:
        *   For each used vehicle (USV/MSB): An ordered list of visited location IDs (depots, sites, rendezvous points).
        *   For each used vehicle: A detailed schedule including arrival time, service start time (if applicable), service end/departure time for each location in its route.
        *   For MSB-VRP: Explicit indication of deployment/recovery events (which USV, where, when).
        *   Summary metrics: Total cost, makespan, number of vehicles used, individual vehicle utilization (time/distance traveled vs. capacity).
*   **FR4: Configuration:**
    *   Allow configuration of the objective function.
    *   Allow configuration of solver time limits and potentially other relevant solver parameters (e.g., number of workers).

**4.2. Classic MDVRP Solver (Time-Dependent)**

*   **FR5: Solver Implementation:** Utilize the Google OR-Tools **CP-SAT** solver.
*   **FR6: Model Formulation:**
    *   Represent depots and sites as nodes.
    *   Variables for vehicle assignments (`x[i, j, v]`), arrival times (`t[loc, v]`), departure times (`d[loc, v]`).
    *   Constraints:
        *   Site Coverage: Each site visited exactly once.
        *   Flow Conservation: Vehicle flow in = flow out for sites.
        *   Depot Start/End: Each used vehicle starts and ends at a valid depot. Handle potentially unused vehicles correctly.
        *   Endurance: Total time/distance/energy consumed by a vehicle <= capacity.
        *   **Time Propagation (CRITICAL):** Implement robust constraints linking departure time (`d[i, v]`) to arrival time (`t[j, v]`) using the time-dependent travel duration obtained via `calculate_travel` (likely involving `AddElement` constraints based on departure time intervals).
        *   Objective Function: Minimize makespan or total cost (sum of time-dependent travel costs + operational costs).
*   **FR7: Bug Fix (CRITICAL):** Resolve the persistent infeasibility issue observed in the provided example code (`INFEASIBLE: 'var #... as empty domain after intersecting with []'`). This likely involves meticulous checking of variable domains, `AddElement` list/index consistency, constraint interactions (especially time propagation vs. start/end constraints), and capacity limits. Ensure the model is consistently feasible for reasonable inputs.

**4.3. Mobile Support Boat (MSB-VRP) Solver**

*   **FR8: Solver Strategy:**
    *   Acknowledge that OR-Tools `Routing` library is unsuitable.
    *   Prioritize a **Heuristic/Metaheuristic approach** (e.g., Large Neighborhood Search (LNS), Adaptive LNS, Genetic Algorithm, Tabu Search) due to complexity.
    *   Alternatively, explore a **complex CP-SAT formulation** if the heuristic approach proves insufficient, accepting potentially longer solve times and increased modeling complexity.
    *   *Discourage* a pure MILP approach unless instance sizes are guaranteed to be very small.
*   **FR9: Heuristic Approach Details (if chosen):**
    *   *Decomposition Strategy:* Define steps like:
        1.  Initial Clustering/MSB Route Sketching: Group sites or define potential rendezvous zones/rough routes for MSBs.
        2.  USV Sub-problem Solving: For given MSB legs/rendezvous pairs, solve smaller VRPs for the USVs deployed/recovered (potentially using OR-Tools CP-SAT or Routing library for these *sub-problems* if rendezvous points/times are fixed temporarily).
        3.  Iterative Improvement (e.g., LNS): Destroy parts of the combined MSB/USV solution (e.g., remove some sites/USV routes, modify an MSB leg) and rebuild/re-optimize iteratively.
    *   *Integrated Metaheuristic:* Define a solution representation encoding both MSB and USV routes/schedules and develop operators to explore the solution space.
*   **FR10: CP-SAT Approach Details (if chosen):**
    *   Model Formulation must include:
        *   Variables for MSB routes, USV routes, site-to-USV assignment, USV-task-to-MSB-leg assignment, rendezvous points/times.
        *   Constraints for standard routing (coverage, flow, capacity).
        *   **Synchronization Constraints:** MSB arrival time = USV arrival/departure time at rendezvous points +/- deployment/recovery time.
        *   MSB Capacity Constraints (number of USVs carried).
        *   Endurance constraints for both MSBs and USVs.
*   **FR11: Rendezvous Handling:** Define how rendezvous points are determined (pre-defined, chosen from sites, optimized as continuous locations?). Define the time cost associated with deployment/recovery.

**5. Non-Functional Requirements**

*   **NFR1: Performance:** Solve moderately sized problems (e.g., <10 USVs, <50 sites, <3 MSBs) within a reasonable timeframe (target < 5-10 minutes, configurable). Performance will degrade with complexity, especially for MSB-VRP.
*   **NFR2: Scalability:** The solution should be architected to handle increasing numbers of vehicles and sites, although performance limits are expected. The MSB-VRP solver will likely be the bottleneck.
*   **NFR3: Usability:** Provide a clear Python API with type hinting and docstrings. Include example usage scripts. Provide meaningful error messages.
*   **NFR4: Accuracy:** Prioritize feasible solutions. For the classic problem, aim for near-optimal solutions via CP-SAT. For MSB-VRP, heuristic solutions should be high-quality but may not be provably optimal. Travel time calculation accuracy depends on the input forecast quality.
*   **NFR5: Maintainability:** Code should be modular, well-commented, and include unit/integration tests where appropriate.
*   **NFR6: Robustness:** Handle edge cases gracefully (e.g., no feasible solution, trivial inputs like no sites).

**6. Data Model (Summary)**

*   **Inputs:** See FR1. Geospatial coordinates (Lat/Lon standard). Time units consistent (e.g., minutes). Speed units consistent (e.g., knots). Weather effect model (e.g., speed factor per time interval).
*   **Outputs:** See FR3. Route as sequence of location IDs. Schedule as list of (LocationID, ArrivalTime, DepartureTime) tuples. Rendezvous details (LocationID, Time, MSB_ID, USV_ID, Action[Deploy/Recover]).

**7. Technical Considerations**

*   **Language:** Python (>= 3.9 recommended).
*   **Core Library:** Google OR-Tools (latest stable version). Primarily `ortools.sat.python.cp_model`.
*   **Environment:** Use `uv` or `pip` with `requirements.txt` or `pyproject.toml`.
*   **Geospatial Calculations:** Initial implementation can use simplified distance (e.g., Euclidean). Consider future integration with libraries like `geopy` or `shapely` for more accurate great-circle distances if required.
*   **Modularity:** Separate data loading, travel calculation, model building (for each problem type), solving, and output processing into distinct modules/classes.

**8. Future Considerations**

*   Real-time / Dynamic Replanning capabilities.
*   More sophisticated cost models (detailed fuel consumption).
*   Stochastic modeling (uncertainty in travel times, survey durations, weather).
*   Graphical User Interface (GUI).
*   Support for heterogeneous fleets with more complex capabilities.
*   Inclusion of charging constraints/opportunities.
*   Multi-objective optimization Pareto fronts.

**9. Open Questions / Assumptions**

*   Assumption: Weather forecasts are deterministic and accurate for the planning horizon.
*   Assumption: USVs/MSBs travel at specified speeds between points (ignoring acceleration/deceleration phases for now).
*   Assumption: Sites can be visited at any time unless specific time windows are added as inputs later.
*   Question: What is the precise definition and relative importance of cost components (time vs. fuel vs. crew)?
*   Question: Is depot selection fixed per vehicle, or can the solver choose the best start/end depot? (Current example implies fixed start=0, flexible end).
*   Question: For MSB-VRP, can MSBs act as mobile depots where USVs wait, or are rendezvous strictly for deploy/recover actions?