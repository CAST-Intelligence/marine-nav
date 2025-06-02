# USV Mission-Specific Animations

This document outlines various animation concepts that could be added to the Darwin Harbor deck.gl visualization to indicate different USV mission types.

## Mission Animation Concepts

### 1. Hydrographic Survey

#### Vessel-Emanating Animations

- **Multi-beam Sonar Fan**: Triangular sector emanating from vessel bottom
  - Implementation:
    - SolidPolygonLayer with fan shape (60-120° wide) extending from vessel
    - Pulse effect with opacity animation (0.7 → 0.3 → 0.7)
    - Fixed distance relative to vessel size (scales with zoom)
  - Color: Blue gradient fading with distance
  - Behavior: Follows vessel orientation, pointing downward
  - Interaction: Can be toggled and width/range adjusted

- **Side-scan Sonar Beams**: Dual lateral beams extending from vessel sides
  - Implementation:
    - Dual SolidPolygonLayers with long, narrow fan shapes (20-30° each)
    - Independent left/right beams extending perpendicular to vessel heading
    - Animated pulse effect (brightness oscillation) traveling outward
    - Distance tied to vessel size and current zoom level
  - Color: Cyan/light blue with semi-transparency (opacity 0.3-0.5)
  - Behavior: Automatically adjusts angle based on vessel heading
  - Control: Can enable/disable each side independently

- **Sonar Ping Animation**: Expanding concentric circles beneath the vessel
  - Implementation:
    - Multiple ScatterplotLayers with radius animation
    - Create new layer at regular intervals (1-3 seconds)
    - Each expands outward and fades over 2-5 seconds
    - Layer removal once fully expanded/faded
  - Color: White fading to transparent blue
  - Pattern: Repeating sequence during movement
  - Configuration: Adjustable ping frequency and expansion rate

#### Directed Sonar Pulse (Forward/Aft)

- **Forward-looking Sonar**: Narrow beam projecting ahead of vessel
  - Implementation:
    - SolidPolygonLayer with narrow cone (15-30° wide)
    - Extended range (2-3x vessel length)
    - Pulsing animation (ripples within cone)
    - Brightness varies with simulated seafloor features
  - Color: White to blue gradient with subtle oscillation
  - Direction: Always aligned with vessel bow
  - Usage: Primary navigation/obstacle detection visualization

- **Aft Survey Sonar**: Wide beam trailing behind vessel
  - Implementation:
    - SolidPolygonLayer with wide fan (90-120°)
    - Extends from vessel stern
    - Gradual fade-in of seafloor "results" (colored polygons)
    - Incorporates swath coverage indication
  - Color: Blue base with multi-colored "results" overlaid
  - Coverage: Leaves persistent "surveyed area" trail
  - Integration: Can display simulated depth readings

- **Sonar Sweep Lines**: Parallel lines behind the vessel that indicate survey coverage
  - Implementation:
    - Multiple PathLayers generated dynamically during movement
    - Spacing based on simulated sonar coverage width
    - Persistence after vessel passes
    - Fades over specified time/distance
  - Color: Blue gradient fading out with distance
  - Pattern: Creates "lawn mower" pattern during survey
  - Coverage: Visual indicator of surveyed vs. unsurveyed areas

- **Bathymetric Depth Grid**: Color-coded grid cells that get "filled in" as the vessel passes
  - Implementation:
    - GridCellLayer with cells that update when vessel sonar passes over
    - Progressive reveal of simulated seafloor data
    - Cell size adjustable based on survey resolution
    - Permanent or temporary display modes
  - Color: Blue-to-red gradient based on simulated depth
  - Persistence: Maintains complete survey coverage visualization
  - Integration: Could incorporate real bathymetry data when available

### 2. Environmental Monitoring

#### Vessel-Emanating Animations

- **Water Sampling Jets**: Directional sampling animations showing water collection
  - Implementation:
    - Multiple LineLayer instances with animated length/opacity
    - Originate from vessel sides/bottom
    - Curl inward to simulate water being drawn into sensors
    - Triggered at configurable intervals or specific locations
  - Color: Transparent blue with white highlights
  - Frequency: Regular intervals or at designated sampling points
  - Visual cues: Small flash when sample "captured"

- **Sensor Array Visualization**: Radial pattern showing active sensors
  - Implementation:
    - CircleLayer with multiple small circles arranged around vessel
    - Each circle represents different sensor type
    - Subtle pulsing animation for active sensors
    - Color changes to indicate reading intensity
  - Color: Different colors for different sensor types (temperature, pH, turbidity, etc.)
  - Pattern: Organized radial arrangement around vessel
  - Interaction: Hover for sensor type/reading (potential future enhancement)

- **Data Collection Aura**: Ambient glow indicating active monitoring
  - Implementation:
    - ScatterplotLayer with large radius centered on vessel
    - Very low opacity (0.05-0.15)
    - Subtle color shifts based on aggregate readings
    - Size scales with vessel and zoom level
  - Color: Teal/green base with shifts toward yellow/red for anomalous readings
  - Behavior: Constant presence during monitoring missions
  - Purpose: Indicates active environmental monitoring status

- **Water Quality Sampling Pulses**: Periodic emanating rings with data indicators
  - Implementation:
    - ScatterplotLayer with radius animation triggered at intervals
    - Text or icon elements appear briefly at pulse edge showing readings
    - Multiple pulses can be active simultaneously
    - Each expands to configurable distance then fades
  - Color: Green/blue with color shift based on readings
  - Timing: Regular intervals (configurable sampling rate)
  - Data visualization: Brief display of actual or simulated readings

#### Directed Sampling Systems

- **Forward Environmental Scanner**: Narrow beam for upcoming water mass analysis
  - Implementation:
    - SolidPolygonLayer with narrow triangular shape
    - Extends ahead of vessel in direction of travel
    - Contains animated particle flow (small dots moving toward vessel)
    - Subtle color variations representing predicted conditions
  - Color: Base cyan with variable color overlay based on conditions
  - Purpose: Predictive sampling of approaching water conditions
  - Configuration: Adjustable range and analysis parameters

- **Trailing Sample Collection**: Aft-directed sampling system visualization
  - Implementation:
    - Multiple LineLayer instances creating "wake sampling" effect
    - Lines curve from wake into vessel
    - Animated dots traveling along lines represent samples
    - Small flash indicators when sample processed
  - Color: White/blue with sample indicators in various colors
  - Pattern: Follows behind vessel with slight randomization
  - Integration: Could display actual or simulated result values

- **Water Quality Visualization**: Trailing color indicators showing measurements
  - Implementation:
    - SolidPolygonLayer with dynamic coloring following vessel
    - Width based on sensor coverage
    - Color transitions based on readings
    - Persistence configurable (temporary or permanent trail)
  - Color: Green to red gradient based on simulated water quality
  - Coverage: Creates "environmental map" of traversed area
  - Data: Could incorporate real data when available

- **Deployed Sensors**: Small marker buoys dropped at intervals
  - Implementation:
    - IconLayer with persistent markers left along path
    - Deployment animation (brief expansion effect)
    - Connection lines to vessel until specific distance reached
    - Optional "active" animation (pulsing) after deployment
  - Visuals: Small buoy icons that remain after vessel passes
  - Deployment: Triggered manually or at predetermined intervals/locations
  - Network: Could show connection lines between deployed sensors

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