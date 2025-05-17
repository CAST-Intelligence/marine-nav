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

## Web Visualization - DataStar with PixiJS and Leaflet

### Recommended Patterns

- **DataStar Signal Patterns**:
  - Use DataStar signals directly as function parameters, not through DOM attributes
  - Example: `updateMap($lat, $lng)` rather than accessing values from the DOM
  - Avoid using `document.body.getAttribute('data-lat')` - use signal variables instead

- **PixiJS + Leaflet Integration**:
  - Initialize map and PixiOverlay separately from drawing logic
  - Store utility functions (project, scale) globally for use outside the PixiOverlay callback
  - Create separate draw functions that can be called from signal handlers

### Example Structure

```javascript
// Global variables to access across functions
let map, pixiOverlay, circle;
let project, scale;  // Store PixiOverlay utility functions

// Initialize the map and PixiOverlay
window.setupMap = function(initialLat, initialLng) {
    // Set up map here...
    
    // Set up PixiOverlay with minimal callback
    pixiOverlay = L.pixiOverlay(function(utils) {
        // Store utility functions globally
        project = utils.latLngToLayerPoint;
        scale = utils.getScale();
        
        // Render container
        utils.getRenderer().render(utils.getContainer());
    }, pixiContainer);
    
    // Initial drawing
    drawCircle(initialLat, initialLng);
}

// Separate draw function that can be called from signal handlers
window.drawCircle = function(lat, lng) {
    // Drawing logic here using project and scale
}

// Signal change handler
window.updateMap = function(lat, lng) {
    drawCircle(lat, lng);
    pixiOverlay.redraw();
}
```

### Common Issues

- PixiOverlay functions (project, scale) are only available inside the overlay callback
- DataStar signals should replace any DOM attribute access
- Keep map center fixed while moving overlay elements to create independent movement