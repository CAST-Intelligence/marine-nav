# Hexagonal Grid Hierarchical Routing

## Overview

This document outlines the plan for developing a hex-based version of our grid network path finding algorithm, using H3 indexing and hierarchical routing for improved efficiency. This approach addresses several limitations in the current square grid implementation:

1. **Distance asymmetry** - 8-directional square grid creates uneven distances (1.0 vs 1.414)
2. **Directional bias** - Square grids favor cardinal directions
3. **Computational scaling** - Flat representation becomes inefficient for large spaces
4. **Lack of abstraction** - No mechanism for multi-resolution analysis

## H3 Advantages

The [H3 Geospatial Indexing System](https://h3geo.org/) provides several advantages for our routing application:

- **Uniform movement cost** between adjacent hexagons
- **Built-in hierarchical structure** for multi-resolution routing
- **Efficient spatial operations** (neighbor finding, distance calculation)
- **Global indexing system** for geographic coordinates
- **Cell containment relationships** between resolution levels

## Implementation Approaches

### 1. Spatial Representation Options

#### Option A: Full Geographic Implementation
- Convert problem to geographic coordinates (lat/long)
- Use H3's native geographic indexing
- Benefits: Leverages all H3 functionality, enables real-world applications
- Challenges: Requires mapping abstract problem to geographic space

#### Option B: Abstract Hex Grid
- Implement hexagonal grid in abstract coordinate space
- Adapt H3 concepts but use custom indexing for abstract space
- Benefits: Simpler for non-geographic problems
- Challenges: Loses some H3 functionality, requires custom implementation

### 2. Hierarchical Resolution Strategy

#### Option A: Top-Down Approach
- Start at coarse resolution (large hexagons)
- Find approximate path through large cells
- Recursively refine path in relevant higher-resolution cells
- Benefits: Extremely efficient for large spaces
- Challenges: May miss optimal paths through "boundary" regions

#### Option B: Bottom-Up with Aggregation
- Define problem at finest resolution
- Aggregate costs/constraints to higher levels
- Use higher levels for initial routing, then refine
- Benefits: Preserves optimality better
- Challenges: More complex implementation, still requires fine-resolution data

### 3. CP-SAT Integration Options

#### Option A: Multi-Resolution Model
- Define variables at multiple resolution levels
- Connect variables across levels with constraints
- Benefits: Directly encodes hierarchy in the model
- Challenges: Complex constraint system, potential solver inefficiency

#### Option B: Resolution Refinement
- Solve at coarse resolution first
- Fix coarse path, then solve refined problems within each coarse cell
- Benefits: Series of smaller, more manageable problems
- Challenges: May not find global optimum

#### Option C: Abstraction-Refinement Loop
- Iteratively alternate between coarse and fine resolutions
- Use fine-level results to update coarse model
- Benefits: Can converge toward global optimum
- Challenges: Complex implementation, convergence not guaranteed

### 4. Handling Required Visit Sites

#### Option A: Multi-Resolution Site Representation
- Represent sites at multiple resolution levels
- Route through appropriate resolution based on site density
- Benefits: Adapts to varying site distributions
- Challenges: Defining appropriate resolution for each region

#### Option B: Resolution Boundaries at Required Sites
- Force resolution changes at required visit locations
- Benefits: Ensures precision where needed
- Challenges: Potential inefficiency in sparse areas

### 5. Time-Dependent Costs Implementation

#### Option A: Resolution-Dependent Time Periods
- Different time scales at different resolutions
- Coarse resolution = longer time periods
- Benefits: Matches temporal and spatial scales
- Challenges: Mapping between different temporal resolutions

#### Option B: Universal Time Periods
- Same time periods across all resolution levels
- Higher resolution cells inherit from parent cells
- Benefits: Simpler implementation, consistent temporal model
- Challenges: May not capture fine temporal variations efficiently

## Technical Components

### Core Libraries:
- **h3-py**: Python bindings for Uber's H3 library
- **OR-Tools CP-SAT**: Constraint programming solver
- **GeoPandas**: For geographic data handling (if using geographic approach)

### Visualization:
- **Folium/Leaflet**: Interactive maps (geographic approach)
- **Matplotlib** with custom hexagon rendering
- **h3-matplotlib**: Specialized plotting for H3 hexagons

## Development Roadmap

1. **Prototype**: Basic hex grid with manually defined hierarchy
2. **H3 Integration**: Implement proper H3 indexing and operations
3. **Hierarchical Model**: Develop multi-resolution path finding
4. **CP-SAT Adaptation**: Modify constraints for hexagonal space
5. **Required Sites**: Implement hierarchical required site handling
6. **Time Dependency**: Add time-dependent costs at multiple resolutions
7. **Visualization**: Develop multi-resolution visualization techniques

## Potential Challenges

- **Boundary Effects**: Handling transitions between different resolution levels
- **Constraint Complexity**: Managing connections between hierarchy levels
- **Computational Efficiency**: Balancing precision vs. performance
- **Path Smoothing**: Ensuring natural paths across resolution boundaries
- **Model Size**: Managing problem size for large areas with multiple resolution levels

## Recommended Initial Approach

For our initial implementation, we recommend:

1. Start with **Abstract Hex Grid** (Option B from Spatial Representation)
2. Use **Top-Down Approach** for hierarchical resolution
3. Implement **Resolution Refinement** for CP-SAT integration
4. Use **Multi-Resolution Site Representation** for required visit sites
5. Begin with **Universal Time Periods** for time-dependent costs

This approach balances implementation complexity with the benefits of hierarchical routing, while providing a clear path for future enhancements.

## Next Steps

1. Set up a development environment with h3-py and necessary dependencies
2. Create a simple hex grid representation with basic neighbor relationships
3. Implement a minimal hierarchical structure with 2-3 resolution levels
4. Develop visualization tools for hex grids at multiple resolutions
5. Adapt the CP-SAT model for hexagonal grid constraints