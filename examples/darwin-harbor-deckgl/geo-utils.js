/**
 * Geographic utilities for marine navigation
 * Contains functions for distance calculations, coordinate transformations,
 * path interpolation, and vector field generation
 */

/**
 * Calculate new latitude based on distance and bearing from a starting point
 * @param {number} lat - Starting latitude in decimal degrees
 * @param {number} lng - Starting longitude in decimal degrees
 * @param {number} distanceKm - Distance to travel in kilometers
 * @param {number} bearingDegrees - Direction to travel in degrees (0 = North, 90 = East)
 * @returns {number} - New latitude in decimal degrees
 */
export function calculateNewLatitude(lat, lng, distanceKm, bearingDegrees) {
    // Earth radius in kilometers
    const R = 6371;
    
    // Convert distance and bearing to radians
    const d = distanceKm / R;
    const bearing = bearingDegrees * Math.PI / 180;
    
    // Convert lat/lng to radians
    const lat1 = lat * Math.PI / 180;
    const lng1 = lng * Math.PI / 180;
    
    // Calculate new latitude
    const lat2 = Math.asin(
        Math.sin(lat1) * Math.cos(d) +
        Math.cos(lat1) * Math.sin(d) * Math.cos(bearing)
    );
    
    // Convert back to degrees
    return lat2 * 180 / Math.PI;
}

/**
 * Calculate new longitude based on distance and bearing from a starting point
 * @param {number} lat - Starting latitude in decimal degrees
 * @param {number} lng - Starting longitude in decimal degrees
 * @param {number} distanceKm - Distance to travel in kilometers
 * @param {number} bearingDegrees - Direction to travel in degrees (0 = North, 90 = East)
 * @returns {number} - New longitude in decimal degrees
 */
export function calculateNewLongitude(lat, lng, distanceKm, bearingDegrees) {
    // Earth radius in kilometers
    const R = 6371;

    // Convert distance and bearing to radians
    const d = distanceKm / R;
    const bearing = bearingDegrees * Math.PI / 180;

    // Convert lat/lng to radians
    const lat1 = lat * Math.PI / 180;
    const lng1 = lng * Math.PI / 180;

    // Calculate new latitude first (needed for longitude calculation)
    const lat2 = Math.asin(
        Math.sin(lat1) * Math.cos(d) +
        Math.cos(lat1) * Math.sin(d) * Math.cos(bearing)
    );

    // Calculate new longitude
    const lng2 = lng1 + Math.atan2(
        Math.sin(bearing) * Math.sin(d) * Math.cos(lat1),
        Math.cos(d) - Math.sin(lat1) * Math.sin(lat2)
    );

    // Convert back to degrees and normalize to -180 to +180
    return ((lng2 * 180 / Math.PI) + 540) % 360 - 180;
}

/**
 * Calculate position along a path based on progress
 * @param {Array<Array<number>>} path - Array of [lng, lat] coordinates defining the path
 * @param {number} progress - Value from 0 to 1 indicating progress along the path
 * @returns {Array<number>} - [lng, lat] position at the specified progress point
 */
export function getPositionAlongPath(path, progress) {
    if (!path || path.length < 2) return null;

    // Handle edge cases
    if (progress <= 0) return path[0];
    if (progress >= 1) return path[path.length - 1];

    // Calculate total path length
    let totalDistance = 0;
    const segments = [];

    for (let i = 0; i < path.length - 1; i++) {
        const [lng1, lat1] = path[i];
        const [lng2, lat2] = path[i + 1];

        // Calculate distance between points (simple approximation)
        const distance = Math.sqrt(
            Math.pow(lng2 - lng1, 2) +
            Math.pow(lat2 - lat1, 2)
        );

        segments.push({
            start: path[i],
            end: path[i + 1],
            distance: distance,
            startDist: totalDistance
        });

        totalDistance += distance;
    }

    // Find the target distance along the path
    const targetDistance = totalDistance * progress;

    // Find the segment containing the target position
    let targetSegment = segments[0];
    for (const segment of segments) {
        if (segment.startDist <= targetDistance &&
            segment.startDist + segment.distance >= targetDistance) {
            targetSegment = segment;
            break;
        }
    }

    // Calculate progress within the segment
    const segmentProgress = targetSegment.distance > 0 ?
        (targetDistance - targetSegment.startDist) / targetSegment.distance : 0;

    // Interpolate position
    const [startLng, startLat] = targetSegment.start;
    const [endLng, endLat] = targetSegment.end;

    return [
        startLng + (endLng - startLng) * segmentProgress,
        startLat + (endLat - startLat) * segmentProgress
    ];
}

/**
 * Generate simulated vector field data for the harbor area
 * @param {number} centerLat - Center latitude for the vector field
 * @param {number} centerLng - Center longitude for the vector field
 * @returns {Array<Object>} - Array of vector objects with sourcePosition and targetPosition
 */
export function generateVectorField(centerLat, centerLng) {
    const vectors = [];
    const gridSize = 20; // Number of points in each direction
    const spacing = 0.005; // Smaller spacing for more density

    for (let i = -gridSize/2; i < gridSize/2; i++) {
        for (let j = -gridSize/2; j < gridSize/2; j++) {
            const lat = centerLat + i * spacing;
            const lng = centerLng + j * spacing;

            // Create a circular flow pattern
            const dx = -(lat - centerLat) * 0.01;
            const dy = (lng - centerLng) * 0.01;

            // Normalize for clearer visualization
            const magnitude = Math.sqrt(dx*dx + dy*dy);
            let scaledDx = dx;
            let scaledDy = dy;

            if (magnitude > 0) {
                // Scale to make vectors more visible
                const scaleFactor = 0.005;
                scaledDx = dx/magnitude * scaleFactor;
                scaledDy = dy/magnitude * scaleFactor;
            }

            // Calculate endpoint of the vector
            const endLng = lng + scaledDx;
            const endLat = lat + scaledDy;

            vectors.push({
                sourcePosition: [lng, lat],
                targetPosition: [endLng, endLat]
            });
        }
    }

    console.log(`Generated ${vectors.length} vectors`);
    return vectors;
}