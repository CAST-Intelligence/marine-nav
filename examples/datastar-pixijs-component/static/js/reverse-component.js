// Defines the <reverse-component> custom element

class ReverseComponent extends HTMLElement {

	// --- 1. Observe the 'name' attribute for changes ---
	static get observedAttributes() {
	  return ["name"]; // Corresponds to data-attr-name
	}
  
	// --- 2. Called when the 'name' attribute changes ---
	attributeChangedCallback(attrName, oldValue, newValue) {
	  // Only react if the 'name' attribute actually changed
	  if (attrName === 'name' && oldValue !== newValue) {
		// console.log(`ReverseComponent received name: ${newValue}`); // For debugging
		const reversedValue = (newValue || "").split("").reverse().join("");
  
		// --- 3. Dispatch a custom 'reverse' event with the result ---
		// The detail object contains the payload
		this.dispatchEvent(new CustomEvent("reverse", {
		  detail: { value: reversedValue },
		  bubbles: true, // Allow event to bubble up (good practice)
		  composed: true // Allow event to cross shadow DOM boundaries (if used)
		}));
	  }
	}
  
	// Optional: connectedCallback can be used for initialization if needed
	connectedCallback() {
	  // console.log("ReverseComponent connected to DOM");
	  // If the component had its own internal display, setup would go here.
	}
  }
  
  // --- 4. Register the custom element with the browser ---
  customElements.define("reverse-component", ReverseComponent);
  console.log("reverse-component defined.");