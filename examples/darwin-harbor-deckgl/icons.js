/**
 * Icons for marine navigation visualization
 * Contains utilities for generating SVG icons used in the deck.gl layers
 */

// Cache for icon URLs
let arrowAtlasURL;

/**
 * Create a data URL for the USV icon (red triangle with white circle and ID number)
 * @param {string} id - ID number to display in the vessel icon
 * @returns {string} - Data URL containing the SVG icon
 */
export function createUsvIconAtlas(id = '33') {
    // Only create once and cache for reuse with the same ID
    const cacheKey = `usv-${id}`;
    if (window[cacheKey]) return window[cacheKey];

    // Create SVG for the USV icon (isosceles triangle with numbered circle)
    // Points north (pointy end at top) for proper alignment with the bearing
    const usvSVG = `
        <svg width="128" height="128" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg">
            <!-- Red isosceles triangle (narrower, more pointed) -->
            <!-- Initial position pointing NORTH (upward) -->
            <polygon points="64,0 96,112 32,112" style="fill:#FF0000;stroke-width:0" />
            <!-- White circle with ID -->
            <circle cx="64" cy="70" r="10" style="fill:#FFFFFF;stroke-width:0" />
            <!-- ID number vertically centered in the circle -->
            <!-- <text x="64" y="83" font-family="Arial" font-size="25" font-weight="bold" text-anchor="middle" dominant-baseline="middle" style="fill:#000000;stroke-width:0">${id}</text> -->
        </svg>
    `;

    const iconURL = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(usvSVG)}`;
    window[cacheKey] = iconURL;
    return iconURL;
}

/**
 * Create a data URL for the arrow icon used in vector fields
 * @returns {string} - Data URL containing the SVG arrow icon
 */
export function createArrowAtlas() {
    // Only create once and cache for reuse
    if (arrowAtlasURL) return arrowAtlasURL;

    const arrowSVG = `
        <svg width="128" height="128" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg">
            <polygon points="64,0 128,128 64,96 0,128" style="fill:#4287f5;stroke-width:0" />
        </svg>
    `;

    arrowAtlasURL = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(arrowSVG)}`;
    return arrowAtlasURL;
}