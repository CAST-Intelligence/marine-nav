# USV Mission-Specific Animations

This document outlines various animation concepts that could be added to the Darwin Harbor deck.gl visualization to indicate different USV mission types.

## Mission Animation Concepts

### 1. Hydrographic Survey

- **Sonar Sweep Lines**: Parallel lines behind the vessel that indicate survey coverage
  - Implementation: PathLayer with dynamic path generation based on vessel movement
  - Color: Blue gradient fading out with distance
  
- **Bathymetric Depth Grid**: Color-coded grid cells that get "filled in" as the vessel passes
  - Implementation: GridCellLayer with opacity/color that updates based on vessel position
  - Color: Blue-to-red gradient based on simulated depth

- **Sonar Ping Animation**: Expanding concentric circles beneath the vessel
  - Implementation: ScatterplotLayer with radius animation
  - Color: White fading to transparent

### 2. Environmental Monitoring

- **Water Sampling Animation**: Periodic expanding circles with data collection indicators
  - Implementation: ScatterplotLayer with radius animation triggered at intervals
  - Color: Green/blue with timed display

- **Water Quality Visualization**: Trailing color indicators showing simulated measurements
  - Implementation: SolidPolygonLayer with dynamic coloring
  - Color: Green to red gradient based on simulated water quality 

- **Deployed Sensors**: Small marker buoys dropped at intervals
  - Implementation: IconLayer with persistent markers left along path
  - Visuals: Small buoy icons that remain after vessel passes

### 3. Search and Rescue

- **Spotlight Cone**: Wide detection cone in front of vessel
  - Implementation: SolidPolygonLayer with fan shape wider than radar
  - Color: Yellow with low opacity

- **Search Pattern Grid**: Overlay showing planned search route
  - Implementation: PathLayer showing predetermined search pattern
  - Visuals: Dashed lines showing future path, solid for completed segments

- **Detection Probability Heat Map**: Coverage intensity that builds up in searched areas
  - Implementation: HeatmapLayer that intensifies with vessel presence
  - Color: Blue (low) to red (high) showing search thoroughness

### 4. Autonomous Fishing/Resource Collection

- **Collection Equipment Visualization**: Nets or collection devices trailing vessel
  - Implementation: PathLayer with dynamic width behind vessel
  - Visuals: Net-like pattern with animated collection

- **Resource Indicators**: Periodic catch animations
  - Implementation: IconLayer with pop-up animations at intervals
  - Visuals: Small resource icons appearing when "catch" occurs

- **Capacity Indicators**: Gauge showing current collection amounts
  - Implementation: Custom UI element linked to collection simulation
  - Visuals: Filling bar or gauge with current/maximum capacity

### 5. Border Security/Patrol

- **Security Perimeter**: Moving boundary that follows vessel
  - Implementation: PolygonLayer with dynamic updates
  - Color: Red/orange semi-transparent boundary

- **Target Identification**: Markers for detected vessels or objects
  - Implementation: IconLayer with different icons for different target types
  - Visuals: Color-coded markers with status indicators

- **Communication Range**: Concentric rings showing transmission capability
  - Implementation: ScatterplotLayer with fixed radius
  - Color: Blue fading outward showing signal strength

### 6. Oceanographic Research

- **Instrument Deployment**: Animation showing equipment being lowered
  - Implementation: PathLayer with vertical lines at sampling points
  - Visuals: Animated line extending downward with instrument icon

- **Data Collection Visualization**: Colored indicators of measurements
  - Implementation: TextLayer with numeric readouts at measurement points
  - Visuals: Small data point markers with pop-up values

- **Water Column Profiling**: Vertical slice visualization
  - Implementation: Custom layer showing depth vs. parameter values
  - Color: Multi-colored gradient indicating various measurements with depth

### 7. Mine Detection/Countermeasures

- **Side-Scan Sonar**: Angular sonar beams from vessel sides
  - Implementation: Multiple SolidPolygonLayers with triangular shapes
  - Color: Yellow/orange with pulsing opacity

- **Detection Probability**: Multi-directional sensor coverage
  - Implementation: HeatmapLayer with intensity based on sensor simulation
  - Color: Green to red gradient showing detection likelihood

- **Target Neutralization**: Animation when objects are found
  - Implementation: ScatterplotLayer with expanding/contracting animation
  - Visuals: Concentric circles with "neutralized" icon

### 8. Precision Navigation Training

- **Virtual Waypoints**: Checkpoints that change state when reached
  - Implementation: ScatterplotLayer with color change on proximity
  - Color: Red (unreached) to green (reached)

- **Accuracy Metrics**: Visual indicators of navigation precision
  - Implementation: PathLayer showing ideal vs. actual path
  - Color: Green (ideal) and blue (actual) with deviation highlighting

- **Maneuver Scoring**: Visual feedback on navigation performance
  - Implementation: TextLayer with score display at key points
  - Visuals: Numeric scores with color-coding for performance

## Implementation Approach

To implement these animations, we would:

1. Add mission type selection to the UI (dropdown or toggle buttons)
2. Create a mission-specific animation component for each type
3. Integrate with the existing vessel position and path tracking
4. Add appropriate controls for specific mission parameters

Each animation would primarily use deck.gl layers (LineLayer, ScatterplotLayer, PolygonLayer, etc.) following the existing pattern of the radar visualization. The animations would scale with zoom level using the same approach as the radar animation.

The modular design should allow for easy switching between mission types and combining multiple mission animations when appropriate.