// Defines the <pixi-mangled-text> custom element

// Constants for canvas - keep consistent if needed elsewhere
const canvasWidth = 480; // Adjusted slightly to fit common layouts
const canvasHeight = 80; // Height for text display

class PixiMangledTextComponent extends HTMLElement {
    #pixiApp = null;
    #pixiText = null;
    #initialized = false;

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        // Basic styling for the component host itself within the shadow DOM
        const style = document.createElement('style');
        style.textContent = `
            :host { display: block; width: ${canvasWidth}px; height: ${canvasHeight}px; }
            canvas { border: 1px dashed #ccc; } /* Style canvas */
        `;
        this.shadowRoot.appendChild(style);
    }

    async connectedCallback() {
        if (this.#initialized) return;
        this.#initialized = true;
        // console.log("PixiMangledTextComponent connected");

        this.#pixiApp = new PIXI.Application();
        try {
            await this.#pixiApp.init({
                width: canvasWidth,
                height: canvasHeight,
                backgroundColor: 0xf8f8f8, // Very light grey background
                antialias: true,
                autoDensity: true, // Adjust resolution for device automatically
                resolution: window.devicePixelRatio || 1,
            });
            this.shadowRoot.appendChild(this.#pixiApp.canvas);

            // --- Create the Text Object ---
            this.#pixiText = new PIXI.Text({
                text: this.getAttribute('text-content') || '', // Initial text
                style: {
                    fontFamily: 'Courier New, monospace', // Monospace looks good for mangled
                    fontSize: 16,
                    fill: 0x228B22, // Forest Green
                    align: 'left',
                    wordWrap: true, // Enable word wrapping
                    wordWrapWidth: canvasWidth - 20, // Wrap within canvas width (with padding)
                    breakWords: true, // Break long words if necessary
                }
             });

            // Center text vertically (approximate)
            this.#pixiText.anchor.set(0, 0.5);
            this.#pixiText.x = 10; // Left padding
            this.#pixiText.y = canvasHeight / 2;

            this.#pixiApp.stage.addChild(this.#pixiText);

            // Initial render based on attribute if present
            this._updateText(this.getAttribute('text-content'));

        } catch (error) {
            console.error("Failed to initialize PixiJS in text component:", error);
            this.shadowRoot.innerHTML = `<p style="color: red;">Error initializing PixiJS canvas.</p>`;
        }
    }

    disconnectedCallback() {
        // console.log("PixiMangledTextComponent disconnected - cleaning up");
        if (this.#pixiApp) {
            this.#pixiApp.destroy(true); // Cleanup resources
            this.#pixiApp = null;
            this.#pixiText = null;
            this.#initialized = false;
        }
    }

    static get observedAttributes() {
        return ['text-content']; // Observe the attribute Datastar will set
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (name === 'text-content' && oldValue !== newValue && this.#pixiApp && this.#pixiText) {
            // console.log(`PixiMangledTextComponent received text: ${newValue}`); // Debug
            this._updateText(newValue);
        }
    }

    // Internal method to update the Pixi Text object
    _updateText(newText) {
        if (this.#pixiText) {
             // Handle null/undefined defensively
            this.#pixiText.text = newText || '';
            // Optional: Re-center if text height changes drastically,
            // but for single line updates, setting y once might be enough.
            // this.#pixiText.y = canvasHeight / 2;
        }
    }
}

customElements.define('pixi-mangled-text', PixiMangledTextComponent);
console.log("pixi-mangled-text component defined.");