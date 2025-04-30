# USV Mission Route Planning

Algorithm prototyping for USV (Unmanned Surface Vehicle) Mission Route Planning, with a focus on current-aware navigation and optimized search patterns.

## Features

- Spatio-temporal vector field representation of time-varying marine currents
- Current-aware path planning with physically accurate vector calculations
- Time-dependent constraint handling (energy, capacity)
- Implementation of standard search patterns:
  - Expanding Square
  - Sector Search
  - Parallel Search (with/without drift compensation)
  - Barrier/Trackline search
- Sophisticated drift detection and simulation
- Visualization tools for time-varying vector fields and navigation grids
- Spatio-Temporal A* (ST-A*) algorithm for planning in dynamic environments

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/CAST-Intelligence/marine-nav.git
   cd marine-nav
   ```

2. Create a virtual environment and install dependencies:
   ```
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

## Usage

The application can be run with various demo modes:

### Vector Field Demo

Shows a sample vector field with currents and obstacles:

```
python examples/vector_field_demo.py
```

### Path Planning Demo

Demonstrates current-aware path planning compared to standard path planning:

```
python examples/path_planning_demo.py
```

### Spatio-Temporal Path Planning Demo

Demonstrates time-dependent path planning with ST-A* algorithm:

```
python examples/spatio_temporal_demo.py
```

## Project Structure

- **src/core/**: Core components like vector field and grid representations
  - **vector_field.py**: Static vector field implementation
  - **spatio_temporal_field.py**: Time-varying vector field with 3D interpolation
  - **grid.py**: Basic navigation grid
  - **temporal_grid.py**: Time-aware grid with accurate physics
- **src/algorithms/**: Path planning and search pattern algorithms
  - **path_planning.py**: Standard A* and search pattern implementations
  - **energy_optimal_path.py**: Energy optimization with drift detection
  - **network_path_finding.py**: Graph-based path planning
  - **spatio_temporal_astar.py**: Time-dependent A* with constraint checking
- **src/visualization/**: Visualization tools
- **src/utils/**: Utility functions
- **examples/**: Example scripts demonstrating functionality
  - **vector_field_demo.py**: Demo of vector field visualization
  - **path_planning_demo.py**: Current-aware path planning comparison
  - **spatio_temporal_demo.py**: Time-dependent planning with ST-A*

## Development

The development plan is outlined in `dev-plan.md`. Current focus areas:

1. Spatio-temporal environment representation (implemented)
2. Single USV with time-dependent optimization (implemented)
3. Multiple USVs with current awareness and coordination
4. Support vessel coordination and mobile depot logistics

## References

- [Google's OR-Tools routing functionality](https://developers.google.com/optimization/routing)
- [USV path planning](https://www.mdpi.com/2077-1312/11/8/1556)
- [Navigational / Search and Rescue Algorithms](https://sites.google.com/site/navigationalalgorithms/sar-search-patterns)
- [Applying Generalised Travelling Salesman Problem (GTSP) algorithms in maritime patrolling missions that require mutual support](https://www.degruyterbrill.com/document/doi/10.24415/9789400604537-012/html?lang=en)

## Search Patterns

The system implements multiple search patterns:

- Expanding Square
- Sector Search (accounts for target drift)
- Parallel Search for single/multiple ships
- Trackline search (anticipated route)
- Barrier search (across a channel)
- Contour Search (follows terrain/bathymetry)

## Key Improvements Made

- Added support for time-varying currents with proper spatio-temporal interpolation
- Implemented physically accurate vector math for current interactions
- Created a true spatio-temporal A* algorithm (ST-A*) with time dimension
- Improved the energy model with cubic power calculations
- Added accurate drift simulation via numerical integration
- Added support for constraint checking (energy, capacity)
- Improved visualization with time-dependent displays

## Future Work

- Implement multi-USV coordination with shared constraints
- Develop stochastic forecasting to account for uncertainty
- Create a full mission planning interface with scenario builder
- Integrate with OR-Tools for complex multi-objective routing problems
- Support mobile depot coordination with rendezvous planning

## Data Sources

For real-world implementations, the system can incorporate data from:

- **OSCAR (Ocean Surface Current Analysis Real-time)**:
  - https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_NRT_V2.0
  - [OSCAR v2 Guide](https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_NRT_V2.0)

- **NOAA Operational Forecast Systems**:
  - https://tidesandcurrents.noaa.gov/models.html

- **HYCOM (HYbrid Coordinate Ocean Model)**:
  - https://www.hycom.org/data/glbv0pt08
  
- **Copernicus Marine Service**:
  - https://marine.copernicus.eu/
   