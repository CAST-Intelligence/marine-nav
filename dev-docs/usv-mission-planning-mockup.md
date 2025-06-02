# USV Mission Planning Sequence and UI Mockup

## Mission Types Identified

From `usv-mission-animations.md:7-299`, we have **10 distinct mission types**:

1. **Hydrographic Survey** - Multi-beam/side-scan sonar, bathymetric mapping
2. **Environmental Monitoring** - Water quality sampling, sensor deployment  
3. **Search and Rescue** - Spotlight detection, search pattern coverage
4. **Autonomous Fishing/Resource Collection** - Collection equipment, capacity tracking
5. **Border Security/Patrol** - Security perimeters, target identification
6. **Oceanographic Research** - Instrument deployment, data collection
7. **Minesweeping** - Mine detection and neutralization operations
8. **Underwater Mining** - Seabed resource extraction and processing
9. **Precision Navigation Training** - Waypoint navigation, accuracy metrics
10. **Cargo Transport** - Container transport, loading/unloading operations
11. **Resupply Operations** - Supply chain management, inventory tracking

## Mission Planning Sequence Outline

Based on `usv-mission-feasibility.md:25-116` and `routing-prd.md:18-48`, here's the proposed mission planning workflow:

### Phase 1: Mission Configuration
1. **Mission Type Selection** - Choose from 10 mission types above
2. **Operational Parameters**:
   - Survey site locations (manual or automated selection)
   - Station-keeping requirements (duration, radius)
   - Fleet composition (USV count, MSB support)
   - Power/endurance constraints
3. **Environmental Data Loading**:
   - Current patterns (Copernicus data)
   - Weather forecasts  
   - Coastal boundaries/obstacles

### Phase 2: Route Optimization
1. **Route Planning Algorithm Selection**:
   - Classic MDVRP (multi-depot vehicle routing)
   - MSB-VRP (mobile support boat scenario)
2. **Optimization Objectives**:
   - Minimize total mission time (makespan)
   - Minimize energy consumption
   - Weighted combination of time/energy
3. **Constraint Validation**:
   - Vehicle endurance limits
   - Survey duration requirements (72 hours per site)
   - Environmental conditions

### Phase 3: Mission Feasibility Analysis
1. **Energy Budget Analysis** (from `ElysiumUSV_PowerPlanner.ipynb`)
2. **Power consumption modeling** (transit vs station-keeping)
3. **Solar generation assessment** (Darwin: ~5.7 kWh/m²/day)
4. **Battery capacity validation** (4000 Wh baseline)

### Phase 4: Visualization & Approval
1. **Interactive mission preview** with deck.gl/MapLibre
2. **Mission-specific animations** based on selected mission type
3. **Timeline and energy profile display**
4. **Mission optimization recommendations**

## Mission-Specific Parameters & Payloads

### 1. **Hydrographic Survey**
**Payloads:**
- Multi-beam Echo Sounder (200 kHz, 400 kHz)
- Side-scan Sonar (100/500 kHz dual frequency)
- Sub-bottom Profiler
- Sound Velocity Probe
- DGPS/RTK positioning system

**Parameters:**
- Survey line spacing (10-50m)
- Survey speed (2-8 knots)
- Overlap percentage (10-50%)
- Maximum depth range (50-1000m)
- Data logging interval (0.1-2.0 seconds)

### 2. **Environmental Monitoring**
**Payloads:**
- CTD (Conductivity/Temperature/Depth) sensor
- Water quality multi-parameter probe
- Weather station
- Hydrocarbon detector
- Fluorometer
- Turbidity sensor

**Parameters:**
- Sampling interval (1-60 minutes)
- Depth profiling range (surface to 100m)
- Data transmission frequency
- Sensor calibration schedule
- Alert thresholds (temperature, pH, dissolved oxygen)

### 3. **Search and Rescue**
**Payloads:**
- High-resolution optical camera
- Thermal imaging camera
- Searchlight array
- Emergency communication relay
- Life raft deployment system

**Parameters:**
- Search pattern type (parallel, expanding square, sector)
- Search area dimensions
- Sweep width (50-500m)
- Detection probability target (90-99%)
- Communication range requirements

### 4. **Autonomous Fishing/Resource Collection**
**Payloads:**
- Fish finder sonar
- Automated fishing gear
- Catch storage containers
- Size/species identification camera
- Preservation system (ice/refrigeration)

**Parameters:**
- Target species selection
- Catch quota limits
- Fishing depth range
- Gear deployment duration
- Storage capacity (kg/m³)

### 5. **Border Security/Patrol**
**Payloads:**
- Long-range surveillance radar
- AIS (Automatic Identification System)
- High-zoom optical camera
- Infrared camera
- Communication interception equipment

**Parameters:**
- Patrol route pattern
- Detection range requirements (1-50 nautical miles)
- Loiter time at checkpoints
- Threat classification criteria
- Response protocols

### 6. **Oceanographic Research**
**Payloads:**
- ADCP (Acoustic Doppler Current Profiler)
- Wave height sensor
- Meteorological sensors
- Water sampling bottles
- Deployable instruments (buoys, gliders)

**Parameters:**
- Measurement frequency
- Instrument deployment depth
- Data collection duration
- Calibration requirements
- Research objectives (currents, waves, chemistry)

### 7. **Minesweeping**
**Payloads:**
- High-frequency side-scan sonar
- Magnetometer
- Mine disposal equipment
- ROV deployment system
- Ground-penetrating sonar

**Parameters:**
- Survey grid resolution (1-10m)
- Detection sensitivity settings
- Classification confidence threshold
- Neutralization method selection
- Safety exclusion zones

### 8. **Underwater Mining**
**Payloads:**
- Remotely Operated Vehicle (ROV)
- Underwater cutting/drilling equipment
- Material collection system
- Seafloor mapping sonar
- Sample processing unit
- Hydraulic dredging system

**Parameters:**
- Target material type (polymetallic nodules, precious metals, rare earth elements)
- Mining depth range (50-6000m)
- Collection rate targets (kg/hour)
- Environmental impact monitoring
- Sediment plume management
- Material grade thresholds

### 9. **Precision Navigation Training**
**Payloads:**
- High-precision GNSS
- Inertial navigation system
- Performance monitoring sensors
- Data logging equipment

**Parameters:**
- Waypoint accuracy tolerance (0.5-5.0m)
- Speed variation limits
- Course-keeping precision
- Performance scoring criteria
- Training scenario complexity

### 9. **Cargo Transport**
**Payloads:**
- Cargo containers (standardized sizes)
- Crane/lifting equipment
- Cargo monitoring sensors
- Stability management system
- Refrigeration unit (for perishables)

**Parameters:**
- Cargo type (dry goods, liquids, hazardous)
- Weight capacity (100-5000 kg)
- Volume capacity (1-50 m³)
- Loading/unloading time
- Special handling requirements
- Route optimization priority (time vs. fuel)

### 10. **Resupply Operations**
**Payloads:**
- Modular supply containers
- Fuel transfer system
- Fresh water tanks
- Emergency supply packages
- Automated dispensing system

**Parameters:**
- Supply type (fuel, water, food, medical, equipment)
- Delivery priority levels (routine, urgent, emergency)
- Inventory tracking requirements
- Consumption rate estimates
- Resupply frequency schedule
- Emergency reserve thresholds

## Updated UI Mockup with Dynamic Parameter Controls

```
┌─────────────────────────────────────────────────────────────────────┐
│ USV Mission Planning Dashboard                                      │
├─────────────────┬───────────────────────────────────────────────────┤
│ Mission Setup   │                                                   │
│ ┌─────────────┐ │                                                   │
│ │Mission Type │ │          Interactive Map                          │
│ │ ☑ Survey    │ │        (deck.gl + MapLibre)                       │
│ │ ☐ Monitor   │ │                                                   │
│ │ ☐ Search    │ │     • Survey sites (clickable)                    │
│ │ ☐ Patrol    │ │     • Current vectors                             │
│ │ ☐ Research  │ │     • Optimal routes                              │
│ │ ☐ Minesweep │ │     • Mission animations                          │
│ │ ☐ Mining    │ │     • Payload coverage visualization              │
│ │ ☐ Training  │ │                                                   │
│ │ ☐ Fishing   │ │                                                   │
│ │ ☐ Cargo     │ │                                                   │
│ │ ☐ Resupply  │ │                                                   │
│ └─────────────┘ │                                                   │
│                 │                                                   │
│ Payload Config  │                                                   │
│ ┌─────────────┐ │                                                   │
│ │Survey       │ │                                                   │
│ │☑ Multi-beam│ │                                                   │
│ │☑ Side-scan │ │                                                   │
│ │☐ Sub-bottom│ │                                                   │
│ │☐ SVP       │ │                                                   │
│ └─────────────┘ │                                                   │
│                 │                                                   │
│ Parameters      │                                                   │
│ Line spacing:   │                                                   │
│ [25m    ] ▲▼   │                                                   │
│ Survey speed:   │                                                   │
│ [4.5kts ] ▲▼   │                                                   │
│ Overlap:        │                                                   │
│ [20%    ] ▲▼   │                                                   │
│                 │                                                   │
│ Fleet Config    │                                                   │
│ USVs: [2] ▲▼   │                                                   │
│ MSBs: [1] ▲▼   │                                                   │
│                 │                                                   │
│ [Plan Mission]  │                                                   │
├─────────────────┼───────────────────────────────────────────────────┤
│ Mission Status  │ Energy & Timeline                                 │
│ Status: Config  │ ┌─────────────────────────────────────────────────┐│
│ Sites: 18       │ │ Mission Duration: 54.2 days                    ││
│ Payload: 85kg   │ │ Energy Budget: 89% used                        ││
│ Power: +15W     │ │ Payload Power: 145W continuous                 ││
│ Route: Pending  │ │ Critical Path: Site 12→15                      ││
│                 │ └─────────────────────────────────────────────────┘│
└─────────────────┴───────────────────────────────────────────────────┘
```

### Key UI Components

1. **Mission Type Selector** - Checkboxes/toggle for 11 mission types
2. **Dynamic Payload Selection** - Mission-specific equipment options
3. **Parameter Controls** - Mission-specific operational settings
4. **Fleet Configuration** - Number controls for USV/MSB counts  
5. **Map Interface** - Interactive site placement and route visualization
6. **Optimization Controls** - Radio buttons for objective selection
7. **Real-time Feasibility** - Live energy/time validation with payload impact
8. **Mission Animations** - Type-specific visualizations (sonar fans, search patterns, cargo operations, etc.)

### Dynamic Parameter Interface

**DataStar Signal Structure:**
```javascript
data-signals="{
  missionType: 'survey',
  selectedPayloads: ['multibeam', 'sidescan'],
  surveySpacing: 25,
  surveySpeed: 4.5,
  overlapPercent: 20,
  totalPayloadWeight: 85,
  totalPayloadPower: 145
}"
```

**Parameter Control Logic:**
```html
<!-- Mission Type triggers payload options -->
<div data-on-signal-change="updatePayloadOptions($missionType)">

<!-- Dynamic parameter controls based on mission type -->
<div data-show="$missionType === 'survey'">
  <label>Line Spacing:</label>
  <input type="range" min="10" max="50" data-bind="surveySpacing">
  <span data-text="$surveySpacing + 'm'"></span>
</div>

<div data-show="$missionType === 'cargo'">
  <label>Cargo Type:</label>
  <select data-bind="cargoType">
    <option value="dry">Dry Goods</option>
    <option value="liquid">Liquids</option>
    <option value="hazmat">Hazardous</option>
  </select>
</div>

<!-- Payload selection updates weight/power calculations -->
<div data-on-signal-change="calculatePayloadImpact($selectedPayloads)">
```

**Mission-Specific Parameter Panels:**

1. **Survey Mode**: Line spacing, speed, overlap, depth range
2. **Cargo Mode**: Cargo type, weight limits, loading time, route priority  
3. **Resupply Mode**: Supply types, delivery priorities, inventory tracking
4. **Patrol Mode**: Pattern type, detection range, loiter time
5. **Environmental Mode**: Sampling intervals, sensor calibration, alert thresholds
6. **Minesweeping Mode**: Detection sensitivity, grid resolution, neutralization methods
7. **Mining Mode**: Target materials, extraction rates, environmental monitoring

## Implementation Notes

This interface uses DataStar's reactive patterns with:
- **Props Down, Events Up Pattern**: Signal values passed to functions as parameters
- **Signal Binding**: `data-bind="signalName"` for camelCase signals
- **Dynamic Controls**: Mission type selection reveals relevant payload and parameter options
- **Real-time Updates**: Payload selection automatically recalculates weight, power consumption, and energy budget impact
- **Mission Animations**: deck.gl visualizations that adapt based on selected mission type and active payloads

The design creates a comprehensive mission planning workflow that leverages all existing algorithms and visualization capabilities while providing intuitive configuration for complex multi-mission scenarios.