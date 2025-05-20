// Particle System Simulation for Ocean Currents
// Based on deck.gl wind example

// Constants for the simulation
const DEFAULT_PARTICLE_COUNT = 5000;
const DEFAULT_PARTICLE_LIFETIME = 80; // in frames
const DEFAULT_PARTICLE_DROP_RATE = 0.005;
const DEFAULT_PARTICLE_SIZE = 2.5;
const DEFAULT_PARTICLE_COLOR = [0, 140, 255, 200]; // Blue with alpha

class ParticleSimulation {
  constructor(props = {}) {
    this.particles = [];
    this.bounds = props.bounds || {
      minLng: 130.8, maxLng: 130.9,
      minLat: -12.5, maxLat: -12.4
    };
    this.numParticles = props.numParticles || DEFAULT_PARTICLE_COUNT;
    this.lifetime = props.lifetime || DEFAULT_PARTICLE_LIFETIME;
    this.dropRate = props.dropRate || DEFAULT_PARTICLE_DROP_RATE;
    this.particleSize = props.particleSize || DEFAULT_PARTICLE_SIZE;
    this.particleColor = props.particleColor || DEFAULT_PARTICLE_COLOR;
    this.initParticles();
  }

  // Initialize particles with random positions within bounds
  initParticles() {
    this.particles = [];
    const { minLng, maxLng, minLat, maxLat } = this.bounds;
    
    for (let i = 0; i < this.numParticles; i++) {
      this.particles.push({
        lng: minLng + Math.random() * (maxLng - minLng),
        lat: minLat + Math.random() * (maxLat - minLat),
        age: Math.floor(Math.random() * this.lifetime),
        size: this.particleSize * (0.5 + Math.random() * 0.5) // Add some size variation
      });
    }
  }

  // Get current data for visualization
  getParticleData() {
    return this.particles.map(p => ({
      position: [p.lng, p.lat],
      radius: p.size,
      color: this.particleColor.map((c, i) => i === 3 ? c * (1 - p.age / this.lifetime) : c) // Fade out with age
    }));
  }

  // Update particle positions based on vector field
  updateParticles(getVector) {
    const newParticles = [];
    const { minLng, maxLng, minLat, maxLat } = this.bounds;
    
    // Update existing particles
    for (const particle of this.particles) {
      // Get vector at particle position
      const vector = getVector(particle.lat, particle.lng);
      if (!vector) continue; // Skip if no vector data
      
      // Move particle
      particle.lng += vector.dx;
      particle.lat += vector.dy;
      particle.age += 1;
      
      // Reset particles that are too old or out of bounds
      if (particle.age >= this.lifetime || 
          particle.lng < minLng || particle.lng > maxLng || 
          particle.lat < minLat || particle.lat > maxLat) {
        // Randomly decide whether to keep the particle (controlled by drop rate)
        if (Math.random() > this.dropRate) {
          // Reset position and age
          particle.lng = minLng + Math.random() * (maxLng - minLng);
          particle.lat = minLat + Math.random() * (maxLat - minLat);
          particle.age = 0;
          newParticles.push(particle);
        }
      } else {
        // Keep the particle
        newParticles.push(particle);
      }
    }
    
    // Add new particles to replace dropped ones
    const numToAdd = this.numParticles - newParticles.length;
    for (let i = 0; i < numToAdd; i++) {
      newParticles.push({
        lng: minLng + Math.random() * (maxLng - minLng),
        lat: minLat + Math.random() * (maxLat - minLat),
        age: 0,
        size: this.particleSize * (0.5 + Math.random() * 0.5)
      });
    }
    
    this.particles = newParticles;
  }
}

// Factory function to create a deck.gl ScatterplotLayer for particles
function createParticleLayer(particleSimulation) {
  return new deck.ScatterplotLayer({
    id: 'particle-layer',
    data: particleSimulation.getParticleData(),
    pickable: false,
    stroked: false,
    filled: true,
    opacity: 0.8,
    radiusUnits: 'pixels',
    getPosition: d => d.position,
    getRadius: d => d.radius,
    getFillColor: d => d.color,
    updateTriggers: {
      getPosition: Date.now() // Force updates on animation
    }
  });
}

// Set up vector field sampler
function setupVectorFieldSampler(vectorData) {
  // Create a spatial index for fast lookup
  const vectorIndex = {};
  
  // Simple grid-based indexing - not the most efficient but works for our example
  const gridSize = 0.005; // Approximately matches our data generation
  
  for (const vector of vectorData) {
    const sourceLng = vector.sourcePosition[0];
    const sourceLat = vector.sourcePosition[1];
    
    // Create grid cell key
    const gridX = Math.floor(sourceLng / gridSize);
    const gridY = Math.floor(sourceLat / gridSize);
    const key = `${gridX},${gridY}`;
    
    // Store vector in grid cell
    if (!vectorIndex[key]) {
      vectorIndex[key] = [];
    }
    vectorIndex[key].push({
      lng: sourceLng,
      lat: sourceLat,
      dx: vector.targetPosition[0] - sourceLng,
      dy: vector.targetPosition[1] - sourceLat
    });
  }
  
  // Return a function that finds the closest vector for a given lat/lng
  return function(lat, lng) {
    const gridX = Math.floor(lng / gridSize);
    const gridY = Math.floor(lat / gridSize);
    
    // Check the current cell and adjacent cells
    for (let x = gridX - 1; x <= gridX + 1; x++) {
      for (let y = gridY - 1; y <= gridY + 1; y++) {
        const key = `${x},${y}`;
        const vectors = vectorIndex[key] || [];
        
        if (vectors.length > 0) {
          // Find closest vector
          let minDist = Infinity;
          let closest = null;
          
          for (const vector of vectors) {
            const dist = Math.sqrt(
              Math.pow(vector.lng - lng, 2) + 
              Math.pow(vector.lat - lat, 2)
            );
            
            if (dist < minDist) {
              minDist = dist;
              closest = vector;
            }
          }
          
          if (closest) {
            return closest;
          }
        }
      }
    }
    
    // No vector found in nearby cells
    return null;
  };
}

// Animation loop controller
class AnimationController {
  constructor(options = {}) {
    this.options = options;
    this.simulation = options.simulation;
    this.vectorSampler = options.vectorSampler;
    this.onFrame = options.onFrame;
    this.isRunning = false;
    this.animationFrameId = null;
  }
  
  start() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.animate();
  }
  
  stop() {
    if (!this.isRunning) return;
    this.isRunning = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }
  
  animate() {
    if (!this.isRunning) return;

    try {
      // Update particle positions
      this.simulation.updateParticles(this.vectorSampler);

      // Call frame callback
      if (this.onFrame) {
        this.onFrame();
      }
    } catch (error) {
      console.warn('Error in animation frame:', error);
    }

    // Continue animation loop
    this.animationFrameId = requestAnimationFrame(this.animate.bind(this));
  }
}