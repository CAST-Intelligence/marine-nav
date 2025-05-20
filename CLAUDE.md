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