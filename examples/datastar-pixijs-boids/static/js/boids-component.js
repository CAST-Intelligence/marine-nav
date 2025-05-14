// Defines the <pixi-boids> custom element for rendering boids

// Canvas dimensions - match server simulation size
const canvasWidth = 800;
const canvasHeight = 600;

class PixiBoidsComponent extends HTMLElement {
    #pixiApp = null;
    #boidsGraphics = [];
    #perceptionRadiusGraphic = null;
    #initialized = false;
    #selectedBoidIndex = -1;
    #showPerceptionRadius = false;
    #trailMode = false;
    #params = {
        alignmentWeight: 1.0,
        cohesionWeight: 1.0,
        separationWeight: 1.5,
        speedLimit: 3.0,
        perceptionRadius: 75.0
    };

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        // Basic styling for the component host itself within the shadow DOM
        const style = document.createElement('style');
        style.textContent = `
            :host { display: block; width: ${canvasWidth}px; height: ${canvasHeight}px; }
            canvas { border: 1px solid #ddd; border-radius: 4px; }
            .controls { 
                position: absolute; 
                bottom: 10px; 
                left: 10px; 
                background: rgba(255,255,255,0.7);
                padding: 5px;
                border-radius: 4px;
                font-family: sans-serif;
                font-size: 12px;
                display: flex;
                gap: 8px;
            }
            button {
                padding: 3px 8px;
                border: 1px solid #ccc;
                background: white;
                border-radius: 3px;
                cursor: pointer;
            }
            button:hover {
                background: #f0f0f0;
            }
        `;
        this.shadowRoot.appendChild(style);
    }

    async connectedCallback() {
        if (this.#initialized) return;
        this.#initialized = true;

        // Initialize PixiJS application
        this.#pixiApp = new PIXI.Application();
        try {
            await this.#pixiApp.init({
                width: canvasWidth,
                height: canvasHeight,
                backgroundColor: 0xf0f0f0,
                antialias: true,
                resolution: window.devicePixelRatio || 1,
            });
            this.shadowRoot.appendChild(this.#pixiApp.canvas);

            // Add UI controls
            const controlsDiv = document.createElement('div');
            controlsDiv.className = 'controls';
            controlsDiv.innerHTML = `
                <button id="toggle-perception">Show Perception</button>
                <button id="toggle-trails">Toggle Trails</button>
            `;
            this.shadowRoot.appendChild(controlsDiv);

            // Add event listeners for UI controls
            controlsDiv.querySelector('#toggle-perception').addEventListener('click', () => {
                this.#showPerceptionRadius = !this.#showPerceptionRadius;
                this._updatePerceptionRadiusVisibility();
            });

            controlsDiv.querySelector('#toggle-trails').addEventListener('click', () => {
                this.#trailMode = !this.#trailMode;
                if (!this.#trailMode) {
                    // Clear trails by resetting background
                    this.#pixiApp.renderer.background.color = 0xf0f0f0;
                } else {
                    // Set semi-transparent background for trails
                    this.#pixiApp.renderer.background.alpha = 0.05;
                }
            });

            // Add graphics for perception radius (initially hidden)
            this.#perceptionRadiusGraphic = new PIXI.Graphics();
            this.#pixiApp.stage.addChild(this.#perceptionRadiusGraphic);
            this.#perceptionRadiusGraphic.visible = false;

            // Initialize with current attributes if they exist
            this._updateFromAttributes();

            // Add interactivity - click to select a boid
            this.#pixiApp.canvas.addEventListener('click', this._handleCanvasClick.bind(this));

        } catch (error) {
            console.error("Failed to initialize PixiJS in boids component:", error);
            this.shadowRoot.innerHTML = `<p style="color: red;">Error initializing PixiJS canvas.</p>`;
        }
    }

    disconnectedCallback() {
        if (this.#pixiApp) {
            this.#pixiApp.destroy(true);
            this.#pixiApp = null;
            this.#boidsGraphics = [];
            this.#perceptionRadiusGraphic = null;
            this.#initialized = false;
        }
    }

    static get observedAttributes() {
        return [
            'boids',
            'alignment-weight',
            'cohesion-weight',
            'separation-weight',
            'speed-limit',
            'perception-radius'
        ];
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue === newValue) return;
        
        if (this.#pixiApp) {
            this._updateFromAttributes();
        }
    }

    _updateFromAttributes() {
        // Update simulation parameters
        const alignmentAttr = this.getAttribute('alignment-weight');
        if (alignmentAttr) {
            this.#params.alignmentWeight = parseFloat(alignmentAttr);
        }

        const cohesionAttr = this.getAttribute('cohesion-weight');
        if (cohesionAttr) {
            this.#params.cohesionWeight = parseFloat(cohesionAttr);
        }

        const separationAttr = this.getAttribute('separation-weight');
        if (separationAttr) {
            this.#params.separationWeight = parseFloat(separationAttr);
        }

        const speedLimitAttr = this.getAttribute('speed-limit');
        if (speedLimitAttr) {
            this.#params.speedLimit = parseFloat(speedLimitAttr);
        }

        const perceptionRadiusAttr = this.getAttribute('perception-radius');
        if (perceptionRadiusAttr) {
            this.#params.perceptionRadius = parseFloat(perceptionRadiusAttr);
            this._updatePerceptionRadius();
        }

        // Update boids if there are any
        const boidsAttr = this.getAttribute('boids');
        if (boidsAttr) {
            try {
                const boids = JSON.parse(boidsAttr);
                if (Array.isArray(boids)) {
                    this._updateBoids(boids);
                }
            } catch (e) {
                console.error("Failed to parse boids attribute:", e);
            }
        }
    }

    _updateBoids(boids) {
        if (!this.#pixiApp) return;

        // If number of boids has changed, recreate all graphics
        if (this.#boidsGraphics.length !== boids.length) {
            // Remove all existing graphics
            this.#boidsGraphics.forEach(g => g.destroy());
            this.#boidsGraphics = [];

            // Create new graphics for each boid
            for (let i = 0; i < boids.length; i++) {
                const graphic = new PIXI.Graphics();
                this.#pixiApp.stage.addChild(graphic);
                this.#boidsGraphics.push(graphic);
            }
        }

        // Update all boid graphics
        boids.forEach((boid, i) => {
            const graphic = this.#boidsGraphics[i];
            
            // Skip if boid hasn't changed position
            if (graphic.x === boid.x && graphic.y === boid.y && graphic.rotation === boid.heading) return;
            
            // Clear previous drawing if not in trail mode
            if (!this.#trailMode) {
                graphic.clear();
            
                // Draw triangular boid shape
                graphic.beginFill(this._getBoidColor(i));
                
                // Draw triangular shape (point in direction of heading)
                const size = 10; // Slightly larger boids for better visibility
                
                // Main triangle body
                graphic.drawPolygon([
                    size, 0,         // Nose
                    -size/2, size/2, // Bottom right
                    -size/2, -size/2 // Bottom left
                ]);
                
                graphic.endFill();
            }
            
            // Update position and rotation
            graphic.x = boid.x;
            graphic.y = boid.y;
            graphic.rotation = boid.heading;
        });

        // Update perception radius visualization if boid is selected
        if (this.#selectedBoidIndex >= 0 && this.#selectedBoidIndex < boids.length) {
            this._updatePerceptionRadius();
        }
    }

    _getBoidColor(index) {
        // Selected boid is highlighted
        if (index === this.#selectedBoidIndex) {
            return 0xFF3300;
        }
        
        // Color based on index to distinguish boids
        const colors = [0x3366FF, 0x33CC33, 0xFF6600, 0x9933FF, 0xFFCC00];
        return colors[index % colors.length];
    }

    _updatePerceptionRadius() {
        if (!this.#perceptionRadiusGraphic || this.#selectedBoidIndex < 0 || this.#selectedBoidIndex >= this.#boidsGraphics.length) return;
        
        const selectedBoid = this.#boidsGraphics[this.#selectedBoidIndex];
        
        this.#perceptionRadiusGraphic.clear();
        this.#perceptionRadiusGraphic.lineStyle(1, 0xFF3300, 0.5);
        this.#perceptionRadiusGraphic.drawCircle(0, 0, this.#params.perceptionRadius);
        
        this.#perceptionRadiusGraphic.x = selectedBoid.x;
        this.#perceptionRadiusGraphic.y = selectedBoid.y;
    }

    _updatePerceptionRadiusVisibility() {
        if (this.#perceptionRadiusGraphic) {
            this.#perceptionRadiusGraphic.visible = this.#showPerceptionRadius && this.#selectedBoidIndex >= 0;
        }
    }

    _handleCanvasClick(event) {
        // Get click position relative to canvas
        const rect = this.#pixiApp.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Find closest boid
        let closestBoid = -1;
        let closestDistance = 20; // Minimum distance for selection
        
        this.#boidsGraphics.forEach((boid, index) => {
            const dx = x - boid.x;
            const dy = y - boid.y;
            const distance = Math.sqrt(dx*dx + dy*dy);
            
            if (distance < closestDistance) {
                closestDistance = distance;
                closestBoid = index;
            }
        });
        
        // Update selection
        this.#selectedBoidIndex = closestBoid;
        
        // Update graphics
        this._updatePerceptionRadius();
        this._updatePerceptionRadiusVisibility();
        
        // Redraw boids to update colors
        const boidsAttr = this.getAttribute('boids');
        if (boidsAttr) {
            try {
                const boids = JSON.parse(boidsAttr);
                if (Array.isArray(boids)) {
                    // Force redraw of all boids to update colors
                    this.#boidsGraphics.forEach(g => g.clear());
                    this._updateBoids(boids);
                }
            } catch (e) {
                console.error("Failed to parse boids attribute:", e);
            }
        }
    }
}

customElements.define('pixi-boids', PixiBoidsComponent);
console.log("pixi-boids component defined.");