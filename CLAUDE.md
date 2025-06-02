# CLAUDE.md - Guide for Marine Navigation USV Mission Planning

## Project Configuration
- **Environment**: Python with UV packaging
- **Data**: Nautical chart data in TXT and compressed file formats
- **Visualization**: QGIS (sample-ausenc.qgz)

## Commands
- Setup: `uv venv && uv pip install -r requirements.txt`
- Run: `uv run main.py`
- Run examples: `uv run examples/path_planning_demo.py`
- Tests: `uv run -m pytest` or `uv run -m pytest tests/test_specific.py -v`
- Lint: `uv run -m ruff check .`
- Format: `uv run -m black .`

## Code Guidelines
- **Imports**: Group standard library, third-party, and local imports
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Types**: Use type hints for all functions and variables
- **Documentation**: Docstrings for all functions and classes (Google style)
- **Error Handling**: Use try/except with specific exceptions
- **Algorithms**: Implement search patterns from README.md
- **Data Processing**: Handle nautical chart data with appropriate libraries

## Web Visualization - DataStar with deck.gl and MapLibre

### Recommended DataStar Patterns

- **Props Down, Events Up Pattern**:
  - Pass signal values down as function parameters from DataStar
  - Example: `updateMap($lat, $lng)` to pass signal values directly
  - Trigger events up with DOM events for DataStar to handle signal updates
  - Avoid directly manipulating data flow outside of DataStar's reactive system

- **Signal Binding for Controls**:
  - For camelCase signal names, use `data-bind="camelCaseName"` instead of `data-bind-camelCaseName`
  - Example: `data-bind="showVectorField"` rather than `data-bind-showVectorField`
  - Regular kebab-case (dash-separated) signals work with either approach
  - Use `data-text="$signalName"` to display signal values in the UI

- **Signal Change Handlers**:
  - Use `data-on-signal-change` attribute to invoke functions when signals change
  - Keep functions pure by using parameters to receive signal values
  - Example: `toggleLayers($showVectorField, $showPath, $showAnimation)`

- **Custom Events Integration**:
  - Use `data-on-customEventName` to listen for custom events
  - Access event data via `evt.detail` in event handlers
  - Example: `<div data-on-terraevent="$terraevt = evt.detail"></div>`
  - Custom events should include `bubbles: true` and `cancelable: true`

### deck.gl + MapLibre Integration

- **Layer Management**:
  - Create separate functions for layer creation and visibility toggling
  - Use `deck.MapboxOverlay` to integrate deck.gl with MapLibre
  - Organize code as reusable layer factories
  - Update layers by creating new instances, then updating deck.gl overlay

### Example Structure

```javascript
// Global variables for map and deck.gl
let map, deckOverlay;
let vesselLayer, pathLayer, vectorFieldLayer;

// Initialize the map and deck.gl overlay
window.setupMap = function(initialLat, initialLng) {
    // Create the MapLibre map
    map = new maplibregl.Map({
        container: 'map',
        style: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
        center: [initialLng, initialLat],
        zoom: 12
    });

    // Create layers
    createLayers(initialLat, initialLng);

    // Create the deck.gl overlay
    deckOverlay = new deck.MapboxOverlay({
        interleaved: true,
        layers: [vesselLayer, pathLayer, vectorFieldLayer]
    });

    // Add overlay to map
    map.addControl(deckOverlay);
}

// Separate function to toggle layer visibility
window.toggleLayers = function(showVectorField, showPath) {
    const layers = [vesselLayer]; // Always show vessel

    if (showVectorField) {
        layers.push(vectorFieldLayer);
    }

    if (showPath) {
        layers.push(pathLayer);
    }

    deckOverlay.setProps({
        layers: layers
    });
}

// Update function for signal changes
window.updateMap = function(lat, lng) {
    vesselLayer = createVesselLayer(lat, lng);
    // Let toggleLayers handle layer updates
}
```

### Important Notes

- For mixed case signal names, you MUST use `data-bind="mixedCaseName"` syntax
- Signal binding with `data-bind-mixedCaseName` won't work for camelCase names
- When updating check boxes from code, use `dispatchEvent(new Event('change'))` to trigger DataStar
- Always keep layer management logic separate from UI interaction logic
- Create pure functions that work with the values passed down from DataStar
- Minimize direct DOM manipulation - let DataStar handle the reactive updates

## Custom Events with DataStar

### Pattern for Third-Party Library Integration

When integrating third-party libraries (like TerraDraw) that emit their own events, use this pattern to bridge them with DataStar:

#### 1. HTML Structure
```html
<!-- Add signal to data-signals -->
<body data-signals="{terraevt: null}">
  <!-- Event listener element -->
  <div id="terraEventListener" data-on-terraevent="$terraevt = evt.detail"></div>
  
  <!-- Display event data -->
  <div data-text="JSON.stringify($terraevt)">Event data will appear here</div>
</body>
```

#### 2. JavaScript Event Bridge
```javascript
// Function to dispatch custom events for DataStar
function dispatchTerraEvent(eventType, eventData) {
    const terraEventListener = document.getElementById('terraEventListener');
    if (terraEventListener) {
        const customEvent = new CustomEvent('terraevent', {
            bubbles: true,
            cancelable: true,
            detail: {
                type: eventType,
                data: eventData,
                timestamp: new Date().toISOString()
            }
        });
        terraEventListener.dispatchEvent(customEvent);
    }
}

// Hook into third-party library events
thirdPartyLibrary.on('select', (id) => {
    dispatchTerraEvent('select', { id: id });
});
```

#### 3. Event Data Structure Best Practices
- Always include event `type` for filtering/routing
- Include `timestamp` for debugging and sequencing
- Structure `data` object based on the specific event needs
- Handle arrays vs single values consistently
- Parse third-party event arguments correctly (use `...args` for debugging)

#### 4. Debugging Third-Party Events
```javascript
// Use this pattern to understand event structure
thirdPartyLibrary.on('eventName', (...args) => {
    console.log('Event args:', args);
    console.log('Args length:', args.length);
    console.log('First arg:', args[0]);
    // Then structure your event handler based on findings
});
```

#### 5. Error Handling
```javascript
// Wrap event listener setup with proper error handling
setTimeout(() => {
    try {
        if (thirdPartyInstance && typeof thirdPartyInstance.on === 'function') {
            thirdPartyInstance.on('eventName', handleEvent);
            console.log('Event listener added successfully');
        }
    } catch (error) {
        console.error('Error setting up event listeners:', error);
    }
}, 100); // Small delay ensures library is fully initialized
```

### TerraDraw Specific Integration

For TerraDraw with MapLibre, use this tested pattern:

```javascript
// Get instance after control is added to map
const drawInstance = draw.getTerraDrawInstance();

// TerraDraw event structure:
// change: [featureIds[], action, undefined]
// finish: [featureId, {mode, action}]
// select/deselect: [featureId]

drawInstance.on('change', (...args) => {
    const featureIds = args[0]; // Array of IDs
    const action = args[1]; // 'create', 'update', 'delete'
    const snapshot = drawInstance.getSnapshot();
    const features = featureIds.map(id => 
        snapshot?.find((feature) => feature.id === id)
    ).filter(Boolean);
    
    dispatchTerraEvent('change', {
        action: action,
        featureIds: featureIds,
        features: features
    });
});
```