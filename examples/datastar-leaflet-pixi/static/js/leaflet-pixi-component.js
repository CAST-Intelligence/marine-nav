// Leaflet.PixiOverlay map component
class LeafletPixiMapComponent extends HTMLElement {
    #map = null;
    #pixiOverlay = null;
    #pixiLayer = null;
    #initialized = false;
    #polygons = [];
    #userSelection = null;
    #selectionGraphics = null;

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        
        // Basic styling for the component host within the shadow DOM
        const style = document.createElement('style');
        style.textContent = `
            :host { display: block; width: 100%; height: 100%; }
            div { width: 100%; height: 100%; }
        `;
        
        // Create map container
        const mapContainer = document.createElement('div');
        mapContainer.id = 'map-container';
        
        this.shadowRoot.appendChild(style);
        this.shadowRoot.appendChild(mapContainer);
    }
    
    connectedCallback() {
        if (this.#initialized) return;
        this.#initialized = true;
        
        // Get initial configuration from attributes
        const center = this._getAttributeAsObject('center') || { lat: 37.7749, lng: -122.4194 }; // Default to San Francisco
        const zoom = Number(this.getAttribute('zoom')) || 13;
        
        // Initialize map
        this.#map = L.map(this.shadowRoot.getElementById('map-container'), {
            center: [center.lat, center.lng],
            zoom: zoom,
            zoomControl: false, // We'll handle zoom in our UI
        });
        
        // Add tile layer - OpenStreetMap
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(this.#map);
        
        // Initialize PixiJS layer
        this._initPixiOverlay();
        
        // Add click event handler
        this.#map.on('click', this._handleMapClick.bind(this));
        
        // Handle attribute changes
        this._updateFromAttributes();
    }
    
    disconnectedCallback() {
        // Clean up resources when component is removed
        if (this.#map) {
            this.#map.remove();
            this.#map = null;
        }
        this.#initialized = false;
    }
    
    static get observedAttributes() {
        return ['center', 'zoom', 'polygons', 'user-selection'];
    }
    
    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue === newValue) return;
        if (this.#map) {
            this._updateFromAttributes();
        }
    }
    
    // Initialize PixiJS overlay
    _initPixiOverlay() {
        const pixiContainer = new PIXI.Container();
        this.#selectionGraphics = new PIXI.Graphics();
        pixiContainer.addChild(this.#selectionGraphics);
        
        this.#pixiOverlay = L.pixiOverlay(utils => {
            const renderer = utils.getRenderer();
            const container = utils.getContainer();
            const project = utils.latLngToLayerPoint;
            const scale = utils.getScale();
            
            // Clear previous drawings
            container.children.forEach(child => {
                if (child !== this.#selectionGraphics) {
                    child.destroy();
                }
            });
            while (container.children.length > 1) {
                container.removeChildAt(1);
            }
            
            // Draw polygons
            this.#polygons.forEach(polygon => {
                const graphics = new PIXI.Graphics();
                
                // Set the line style and fill
                graphics.lineStyle(2 / scale, this._hexToDecimal(polygon.color));
                graphics.beginFill(this._hexToDecimal(polygon.fillColor), 0.3);
                
                // Draw polygon
                if (polygon.points && polygon.points.length > 0) {
                    const firstPoint = project([polygon.points[0].lat, polygon.points[0].lng]);
                    graphics.moveTo(firstPoint.x, firstPoint.y);
                    
                    for (let i = 1; i < polygon.points.length; i++) {
                        const point = project([polygon.points[i].lat, polygon.points[i].lng]);
                        graphics.lineTo(point.x, point.y);
                    }
                    
                    // Close the polygon
                    graphics.lineTo(firstPoint.x, firstPoint.y);
                }
                
                graphics.endFill();
                container.addChild(graphics);
            });
            
            // Draw selection circle if it exists
            this.#selectionGraphics.clear();
            if (this.#userSelection) {
                const center = project([this.#userSelection.center.lat, this.#userSelection.center.lng]);
                
                // Convert meters to pixels at current zoom
                const metersPerPixel = 40075016.686 * Math.abs(Math.cos(this.#userSelection.center.lat * Math.PI / 180)) / Math.pow(2, this.#map.getZoom() + 8);
                const radiusInPixels = this.#userSelection.distance / metersPerPixel;
                
                // Draw circle
                this.#selectionGraphics.lineStyle(3 / scale, 0x0000FF, 0.8);
                this.#selectionGraphics.beginFill(0x0000FF, 0.2);
                this.#selectionGraphics.drawCircle(center.x, center.y, radiusInPixels);
                this.#selectionGraphics.endFill();
                
                // Draw center point
                this.#selectionGraphics.lineStyle(1 / scale, 0x000000, 1);
                this.#selectionGraphics.beginFill(0x0000FF, 1);
                this.#selectionGraphics.drawCircle(center.x, center.y, 5 / scale);
                this.#selectionGraphics.endFill();
            }
            
            renderer.render(container);
        }, pixiContainer);
        
        this.#pixiOverlay.addTo(this.#map);
    }
    
    _handleMapClick(e) {
        // Create or update selection at click location
        const selection = {
            center: {
                lat: e.latlng.lat,
                lng: e.latlng.lng
            },
            distance: this.#userSelection ? this.#userSelection.distance : 200 // Default 200m radius
        };
        
        // Update internal state
        this.#userSelection = selection;
        
        // Redraw overlay
        if (this.#pixiOverlay) {
            this.#pixiOverlay.redraw();
        }
        
        // Dispatch selection change event
        this.dispatchEvent(new CustomEvent('selection-change', {
            detail: selection,
            bubbles: true,
            composed: true
        }));
    }
    
    // Helper to convert hex color to decimal (for PIXI)
    _hexToDecimal(hex) {
        // Remove # if present
        hex = hex.replace(/^#/, '');
        
        // Handle 8-digit hex with alpha
        if (hex.length === 8) {
            // Extract alpha value
            const alpha = parseInt(hex.slice(6, 8), 16) / 255;
            // Convert the RGB part to decimal
            return parseInt(hex.slice(0, 6), 16);
        }
        
        // Regular hex to decimal
        return parseInt(hex, 16);
    }
    
    // Helper to parse JSON attribute
    _getAttributeAsObject(name) {
        const attr = this.getAttribute(name);
        if (!attr) return null;
        
        try {
            return JSON.parse(attr);
        } catch (e) {
            console.error(`Failed to parse ${name} attribute:`, e);
            return null;
        }
    }
    
    // Update component based on attributes
    _updateFromAttributes() {
        // Update center and zoom
        const center = this._getAttributeAsObject('center');
        if (center && this.#map) {
            this.#map.setView([center.lat, center.lng], this.#map.getZoom(), { animate: true });
        }
        
        const zoom = this.getAttribute('zoom');
        if (zoom && this.#map) {
            this.#map.setZoom(Number(zoom), { animate: true });
        }
        
        // Update polygons
        const polygons = this._getAttributeAsObject('polygons');
        if (polygons) {
            this.#polygons = polygons;
            if (this.#pixiOverlay) {
                this.#pixiOverlay.redraw();
            }
        }
        
        // Update user selection
        const selection = this._getAttributeAsObject('user-selection');
        if (selection !== null) {
            this.#userSelection = selection;
            if (this.#pixiOverlay) {
                this.#pixiOverlay.redraw();
            }
        }
    }
    
    // Public API for updating the map externally
    setZoom(zoom) {
        if (this.#map) {
            this.#map.setZoom(Number(zoom), { animate: true });
        }
    }
    
    setCenter(lat, lng) {
        if (this.#map) {
            this.#map.panTo([lat, lng], { animate: true });
        }
    }
    
    redraw() {
        if (this.#pixiOverlay) {
            this.#pixiOverlay.redraw();
        }
    }
}

// Register the custom element
customElements.define('leaflet-pixi-map', LeafletPixiMapComponent);
console.log("leaflet-pixi-map component defined.");