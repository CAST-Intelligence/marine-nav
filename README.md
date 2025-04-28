# USV Mission Route Planning

Algorithm prototyping for USV (Unmanned Surface Vehicle) Mission Route Planning, with a focus on current-aware navigation and optimized search patterns.

## Features

- Vector field representation of marine currents
- Current-aware path planning with asymmetric distance/cost calculations
- Implementation of standard search patterns:
  - Expanding Square
  - Sector Search
  - Parallel Search (with/without drift compensation)
  - Barrier/Trackline search
- Visualization tools for vector fields and navigation grids
- A* pathfinding algorithm adapted for current awareness

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
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
python main.py vector-field-demo
```

### Path Planning Demo

Demonstrates current-aware path planning compared to standard path planning:

```
python main.py path-planning-demo
```

## Project Structure

- **src/core/**: Core components like vector field and grid representations
- **src/algorithms/**: Path planning and search pattern algorithms
- **src/visualization/**: Visualization tools
- **src/utils/**: Utility functions
- **examples/**: Example scripts demonstrating functionality

## Development

The development plan is outlined in `dev-plan.md`. Current focus areas:

1. Current-aware environment representation
2. Single USV with current optimization
3. Multiple USVs with current awareness
4. Support vessel coordination

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

## Future Work

- Implement additional search patterns
- Add time-varying currents
- Optimize multi-USV cooperation
- Integrate with OR-Tools for more complex routing problems
- Support mobile depot coordination


