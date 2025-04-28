# USV Mission Route Planning - Development Plan

## 1. Core Components

### 1.1 Mission Parameters
- USV specifications (range, speed, endurance)
- Sensor characteristics (coverage width, overlap requirements)
- Payload constraints (weight, power consumption)
- Depot locations (fixed ports, mobile support vessels)
- Target areas (points, lines, areas)
- Obstacles and restricted zones
- Current vector field representation

### 1.2 Vector Field Topology Implementation
- Current Modeling:
  - 2D grid with current vectors (magnitude and direction)
  - Critical points identification (stagnation, vortices)
  - Current strength variation mapping
- Distance Matrix Calculation:
  - Asymmetric distances based on current effects
  - Lower costs for traveling with currents
  - Higher costs for traveling against currents
  - Energy consumption modeling based on current vectors
  - Eventually - temnporal changes (i.e. weather forecast)
### 1.3 Algorithm Implementation
- Modified search patterns for current-aware navigation:
  - Expanding Square with current compensation
  - Sector Search with drift adjustment
  - Parallel Search with current-optimized track spacing
  - Contour following with energy efficiency
- Route optimization using OR-Tools with custom distance matrix:
  - TSP solver with asymmetric distance matrix
  - VRP solver with time-dependent travel costs
  - Path smoothing for dynamic currents

### 1.4 Coordination Strategies
- Support vessel positioning based on current patterns
- Multi-USV task allocation with current considerations
- Sensor coverage optimization with drift compensation
- Rendezvous planning accounting for current-influenced travel times

## 2. Implementation Phases

### Phase 1: Current-Aware Environment
- Implement vector field grid representation for currents
- Develop asymmetric distance calculation
- Create current visualization on grid
- Implement basic path planning in vector field

### Phase 2: Single USV with Current Optimization
- Adapt search patterns for current conditions
- Develop energy-efficient route planning
- Optimize sensor coverage with drift compensation
- Implement obstacle avoidance considering currents

### Phase 3: Multiple USVs with Current Awareness
- Implement OR-Tools VRP with custom distance matrix
- Develop coordination algorithms for current conditions
- Optimize fleet positioning based on current patterns

### Phase 4: Support Vessel Coordination in Currents
- Optimize support vessel positioning in current fields
- Implement dynamic depot locations with current predictions
- Create current-aware rendezvous scheduling

## 3. Technical Approach

### 3.1 Data Structures
- Vector field grid for current representation
- Asymmetric graph for route planning
- Current-aware parameter objects

### 3.2 Algorithms
- Vector field topology analysis for current characterization
- Modified A* with current-based cost function
- OR-Tools with custom distance callbacks
- Energy optimization with current consideration
- Genetic algorithms for multi-objective current-aware optimization

### 3.3 Current Modeling
- Periodic time-varying vector fields
- Interpolation between measured/predicted current data points
- Critical points identification (stagnation, vortices)
- Current divergence and curl analysis

### 3.4 Visualization
- Vector field display with magnitude/direction indicators
- Current-influenced path visualization
- Energy efficiency heat maps
- Coverage visualization with drift adjustment

## 4. Evaluation Metrics
- Total energy consumption (not just distance)
- Time to completion with current effects
- Area coverage percentage with drift compensation
- Path efficiency relative to current patterns
- Task completion rate in varying current conditions
- Robustness to current changes