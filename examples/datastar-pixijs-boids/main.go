package main

import (
	"embed"
	_ "embed"
	"fmt"
	"io/fs"
	"log"
	"math"
	"math/rand"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	datastar "github.com/starfederation/datastar/sdk/go"
)

//go:embed index.html
var indexHTML []byte

//go:embed static
var staticFiles embed.FS

// Boid represents a single boid in the simulation
type Boid struct {
	ID        int     `json:"id"`
	X         float64 `json:"x"`
	Y         float64 `json:"y"`
	VelocityX float64 `json:"vx"`
	VelocityY float64 `json:"vy"`
	Heading   float64 `json:"heading"` // in radians
}

// SimulationState holds the entire state of the boids simulation
type SimulationState struct {
	Boids     []Boid
	Width     float64
	Height    float64
	mu        sync.RWMutex
	params    SimulationParams
	lastFrame time.Time
}

// SimulationParams contains the parameters that control boid behavior
type SimulationParams struct {
	AlignmentWeight  float64 `json:"alignmentWeight"`
	CohesionWeight   float64 `json:"cohesionWeight"`
	SeparationWeight float64 `json:"separationWeight"`
	SpeedLimit       float64 `json:"speedLimit"`
	BoidCount        int     `json:"boidCount"`
	PerceptionRadius float64 `json:"perceptionRadius"`
}

// ParamRequest is used for reading parameter changes from client
type ParamRequest struct {
	AlignmentWeight  *float64 `json:"alignmentWeight"`
	CohesionWeight   *float64 `json:"cohesionWeight"`
	SeparationWeight *float64 `json:"separationWeight"`
	SpeedLimit       *float64 `json:"speedLimit"`
	BoidCount        *int     `json:"boidCount"`
	PerceptionRadius *float64 `json:"perceptionRadius"`
}

// Default simulation parameters
const (
	DefaultWidth            = 800
	DefaultHeight           = 600
	DefaultAlignmentWeight  = 1.5
	DefaultCohesionWeight   = 1.0
	DefaultSeparationWeight = 1.5
	DefaultSpeedLimit       = 5180.0 // Higher speed limit
	DefaultBoidCount        = 50
	DefaultPerceptionRadius = 75.0
	SimulationTickRate      = 30 * time.Millisecond // back to ~33 FPS
)

// Global simulation state
var simulation SimulationState

func init() {
	// Initialize random number generator
	rand.Seed(time.Now().UnixNano())

	// Setup initial simulation state
	simulation = SimulationState{
		Boids:  make([]Boid, DefaultBoidCount),
		Width:  DefaultWidth,
		Height: DefaultHeight,
		params: SimulationParams{
			AlignmentWeight:  DefaultAlignmentWeight,
			CohesionWeight:   DefaultCohesionWeight,
			SeparationWeight: DefaultSeparationWeight,
			SpeedLimit:       DefaultSpeedLimit,
			BoidCount:        DefaultBoidCount,
			PerceptionRadius: DefaultPerceptionRadius,
		},
		lastFrame: time.Now(),
	}

	// Initialize boids with random positions and velocities
	for i := 0; i < DefaultBoidCount; i++ {
		simulation.Boids[i] = Boid{
			ID:        i,
			X:         rand.Float64() * DefaultWidth,
			Y:         rand.Float64() * DefaultHeight,
			VelocityX: (rand.Float64()*2 - 1) * DefaultSpeedLimit * 0.5,
			VelocityY: (rand.Float64()*2 - 1) * DefaultSpeedLimit * 0.5,
		}
		// Calculate initial heading based on velocity
		simulation.Boids[i].Heading = math.Atan2(simulation.Boids[i].VelocityY, simulation.Boids[i].VelocityX)
	}
}

func main() {
	// Start simulation goroutine
	go runSimulation()

	// Setup router
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)

	// Serve the main HTML page
	r.Get("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write(indexHTML)
	})

	// Endpoint to stream boid positions
	r.Get("/boids-stream", streamBoids)

	// Endpoint to update simulation parameters
	r.Get("/update-params", updateParams)

	// Serve static files
	staticFS, err := fs.Sub(staticFiles, "static")
	if err != nil {
		log.Fatalf("Failed to create sub filesystem: %v", err)
	}
	r.Handle("/static/*", http.StripPrefix("/static/", http.FileServer(http.FS(staticFS))))

	// Start server
	port := ":8080"
	log.Printf("Boids simulation server starting on http://localhost%s", port)
	if err := http.ListenAndServe(port, r); err != nil {
		log.Fatalf("Error starting server: %v", err)
	}
}

// streamBoids streams the current state of all boids to clients using SSE
func streamBoids(w http.ResponseWriter, r *http.Request) {
	sse := datastar.NewSSE(w, r)

	// Send initial state
	simulation.mu.RLock()
	err := sse.MarshalAndMergeSignals(map[string]interface{}{
		"boids":            simulation.Boids,
		"alignmentWeight":  simulation.params.AlignmentWeight,
		"cohesionWeight":   simulation.params.CohesionWeight,
		"separationWeight": simulation.params.SeparationWeight,
		"speedLimit":       simulation.params.SpeedLimit,
		"boidCount":        simulation.params.BoidCount,
		"perceptionRadius": simulation.params.PerceptionRadius,
	})
	simulation.mu.RUnlock()

	if err != nil {
		log.Printf("Error sending initial boid state: %v", err)
		return
	}

	// Create ticker to send updates at regular intervals
	ticker := time.NewTicker(33 * time.Millisecond) // ~30 FPS
	defer ticker.Stop()

	// Monitor connection and send updates
	for {
		select {
		case <-ticker.C:
			simulation.mu.RLock()
			err := sse.MarshalAndMergeSignals(map[string]interface{}{
				"boids": simulation.Boids,
			})
			simulation.mu.RUnlock()

			if err != nil {
				log.Printf("Error streaming boid state: %v", err)
				return // Connection likely closed
			}
		case <-r.Context().Done():
			log.Println("Client disconnected from boid stream")
			return
		}
	}
}

// Helper functions to parse signal values
func parseFloat64Value(value interface{}) (float64, error) {
	switch v := value.(type) {
	case float64:
		return v, nil
	case int:
		return float64(v), nil
	case string:
		return strconv.ParseFloat(v, 64)
	default:
		return 0, fmt.Errorf("unsupported type for float64 conversion: %T", value)
	}
}

func parseIntValue(value interface{}) (int, error) {
	switch v := value.(type) {
	case int:
		return v, nil
	case float64:
		return int(v), nil
	case string:
		parsed, err := strconv.ParseInt(v, 10, 64)
		return int(parsed), err
	default:
		return 0, fmt.Errorf("unsupported type for int conversion: %T", value)
	}
}

// updateParams reads parameter updates from datastar signals and updates the simulation
func updateParams(w http.ResponseWriter, r *http.Request) {
	// Create a map to receive signals
	signals := make(map[string]interface{})

	// Read signals from request
	if err := datastar.ReadSignals(r, &signals); err != nil {
		log.Printf("Error reading signals: %v", err)
		return
	}

	fmt.Println("Signals:", signals)
	// Prepare param request with pointers
	reqData := &ParamRequest{}

	// Process each parameter individually
	if val, ok := signals["alignmentWeight"]; ok {
		if floatVal, err := parseFloat64Value(val); err == nil {
			reqData.AlignmentWeight = &floatVal
		}
	}
	if val, ok := signals["cohesionWeight"]; ok {
		if floatVal, err := parseFloat64Value(val); err == nil {
			reqData.CohesionWeight = &floatVal
		}
	}
	if val, ok := signals["separationWeight"]; ok {
		if floatVal, err := parseFloat64Value(val); err == nil {
			reqData.SeparationWeight = &floatVal
		}
	}
	if val, ok := signals["speedLimit"]; ok {
		if floatVal, err := parseFloat64Value(val); err == nil {
			reqData.SpeedLimit = &floatVal
		}
	}
	if val, ok := signals["perceptionRadius"]; ok {
		if floatVal, err := parseFloat64Value(val); err == nil {
			reqData.PerceptionRadius = &floatVal
		}
	}
	if val, ok := signals["boidCount"]; ok {
		if intVal, err := parseIntValue(val); err == nil {
			reqData.BoidCount = &intVal
		}
	}

	simulation.mu.Lock()

	// Update parameters only if they were provided in the request
	if reqData.AlignmentWeight != nil {
		simulation.params.AlignmentWeight = *reqData.AlignmentWeight
	}
	if reqData.CohesionWeight != nil {
		simulation.params.CohesionWeight = *reqData.CohesionWeight
	}
	if reqData.SeparationWeight != nil {
		simulation.params.SeparationWeight = *reqData.SeparationWeight
	}
	if reqData.SpeedLimit != nil {
		simulation.params.SpeedLimit = *reqData.SpeedLimit
	}
	if reqData.PerceptionRadius != nil {
		simulation.params.PerceptionRadius = *reqData.PerceptionRadius
	}

	// Handle boid count changes (this requires adding/removing boids)
	if reqData.BoidCount != nil && *reqData.BoidCount != simulation.params.BoidCount {
		newCount := *reqData.BoidCount
		currentCount := len(simulation.Boids)

		if newCount > currentCount {
			// Add new boids
			for i := currentCount; i < newCount; i++ {
				newBoid := Boid{
					ID:        i,
					X:         rand.Float64() * simulation.Width,
					Y:         rand.Float64() * simulation.Height,
					VelocityX: (rand.Float64()*2 - 1) * simulation.params.SpeedLimit * 0.5,
					VelocityY: (rand.Float64()*2 - 1) * simulation.params.SpeedLimit * 0.5,
				}
				newBoid.Heading = math.Atan2(newBoid.VelocityY, newBoid.VelocityX)
				simulation.Boids = append(simulation.Boids, newBoid)
			}
		} else if newCount < currentCount {
			// Remove boids (keep the first newCount boids)
			simulation.Boids = simulation.Boids[:newCount]
		}

		simulation.params.BoidCount = newCount
	}

	simulation.mu.Unlock()

	// Send updated parameters back to clients
	sse := datastar.NewSSE(w, r)
	simulation.mu.RLock()
	err := sse.MarshalAndMergeSignals(map[string]interface{}{
		"alignmentWeight":  simulation.params.AlignmentWeight,
		"cohesionWeight":   simulation.params.CohesionWeight,
		"separationWeight": simulation.params.SeparationWeight,
		"speedLimit":       simulation.params.SpeedLimit,
		"boidCount":        simulation.params.BoidCount,
		"perceptionRadius": simulation.params.PerceptionRadius,
	})
	simulation.mu.RUnlock()

	if err != nil {
		log.Printf("Error sending updated parameters: %v", err)
	}
}

// runSimulation continuously updates the boid simulation state
func runSimulation() {
	ticker := time.NewTicker(SimulationTickRate)
	defer ticker.Stop()

	for range ticker.C {
		now := time.Now()
		simulation.mu.Lock()

		// Calculate time delta in seconds for smoother animation
		deltaTime := now.Sub(simulation.lastFrame).Seconds()
		simulation.lastFrame = now

		// Apply boid behaviors and update positions
		updateBoids(deltaTime)

		simulation.mu.Unlock()
	}
}

// updateBoids applies boid rules and updates positions
func updateBoids(deltaTime float64) {
	// Make copy of the current state to calculate next positions from
	numBoids := len(simulation.Boids)

	// Pre-compute next velocities based on boid rules
	newVelocities := make([][2]float64, numBoids)

	for i := 0; i < numBoids; i++ {
		boid := &simulation.Boids[i]

		// Apply the three boid rules to calculate steering forces
		alignX, alignY := alignment(i)
		cohesionX, cohesionY := cohesion(i)
		separationX, separationY := separation(i)

		// Apply weights to the steering forces
		alignX *= simulation.params.AlignmentWeight
		alignY *= simulation.params.AlignmentWeight

		cohesionX *= simulation.params.CohesionWeight
		cohesionY *= simulation.params.CohesionWeight

		separationX *= simulation.params.SeparationWeight
		separationY *= simulation.params.SeparationWeight

		// Calculate new velocity by adding all forces
		newVX := boid.VelocityX + alignX + cohesionX + separationX
		newVY := boid.VelocityY + alignY + cohesionY + separationY

		// Limit speed
		speed := math.Sqrt(newVX*newVX + newVY*newVY)
		if speed > simulation.params.SpeedLimit {
			newVX = (newVX / speed) * simulation.params.SpeedLimit
			newVY = (newVY / speed) * simulation.params.SpeedLimit
		}

		newVelocities[i] = [2]float64{newVX, newVY}
	}

	// Apply the new velocities and update positions
	for i := 0; i < numBoids; i++ {
		boid := &simulation.Boids[i]

		// Update velocity
		boid.VelocityX = newVelocities[i][0]
		boid.VelocityY = newVelocities[i][1]

		// Update heading if the boid is moving
		speed := math.Sqrt(boid.VelocityX*boid.VelocityX + boid.VelocityY*boid.VelocityY)
		if speed > 0.1 {
			boid.Heading = math.Atan2(boid.VelocityY, boid.VelocityX)
		}

		// Update position
		boid.X += boid.VelocityX * deltaTime
		boid.Y += boid.VelocityY * deltaTime

		// Wrap around screen edges (toroidal world)
		if boid.X < 0 {
			boid.X += simulation.Width
		} else if boid.X >= simulation.Width {
			boid.X -= simulation.Width
		}

		if boid.Y < 0 {
			boid.Y += simulation.Height
		} else if boid.Y >= simulation.Height {
			boid.Y -= simulation.Height
		}
	}
}

// alignment calculates the average velocity of nearby boids
func alignment(index int) (float64, float64) {
	boid := &simulation.Boids[index]
	var avgVX, avgVY float64
	count := 0

	for i, other := range simulation.Boids {
		if i == index {
			continue
		}

		// Calculate distance with wrapping at screen edges
		dx := math.Abs(boid.X - other.X)
		if dx > simulation.Width/2 {
			dx = simulation.Width - dx
		}

		dy := math.Abs(boid.Y - other.Y)
		if dy > simulation.Height/2 {
			dy = simulation.Height - dy
		}

		dist := math.Sqrt(dx*dx + dy*dy)

		// If within perception radius, include in alignment calculation
		if dist <= simulation.params.PerceptionRadius {
			avgVX += other.VelocityX
			avgVY += other.VelocityY
			count++
		}
	}

	if count > 0 {
		// Calculate average and then steering force
		avgVX /= float64(count)
		avgVY /= float64(count)

		// Return steering force (desired - current)
		return avgVX - boid.VelocityX, avgVY - boid.VelocityY
	}

	return 0, 0
}

// cohesion calculates a force to steer towards the center of mass of nearby boids
func cohesion(index int) (float64, float64) {
	boid := &simulation.Boids[index]
	var centerX, centerY float64
	count := 0

	for i, other := range simulation.Boids {
		if i == index {
			continue
		}

		// Calculate distance with wrapping at screen edges
		dx := math.Abs(boid.X - other.X)
		if dx > simulation.Width/2 {
			dx = simulation.Width - dx
		}

		dy := math.Abs(boid.Y - other.Y)
		if dy > simulation.Height/2 {
			dy = simulation.Height - dy
		}

		dist := math.Sqrt(dx*dx + dy*dy)

		// If within perception radius, include in cohesion calculation
		if dist <= simulation.params.PerceptionRadius {
			// Handle wrapping - get position closest to this boid
			otherX := other.X
			otherY := other.Y

			if math.Abs(boid.X-other.X) > simulation.Width/2 {
				if boid.X > other.X {
					otherX += simulation.Width
				} else {
					otherX -= simulation.Width
				}
			}

			if math.Abs(boid.Y-other.Y) > simulation.Height/2 {
				if boid.Y > other.Y {
					otherY += simulation.Height
				} else {
					otherY -= simulation.Height
				}
			}

			centerX += otherX
			centerY += otherY
			count++
		}
	}

	if count > 0 {
		// Calculate center of mass
		centerX /= float64(count)
		centerY /= float64(count)

		// Calculate steering force towards center of mass
		steerX := centerX - boid.X
		steerY := centerY - boid.Y

		// Scale steering force
		steerMag := math.Sqrt(steerX*steerX + steerY*steerY)
		if steerMag > 0 {
			steerX = steerX / steerMag * 0.2 // Increased from 0.05
			steerY = steerY / steerMag * 0.2 // Increased from 0.05
		}

		return steerX, steerY
	}

	return 0, 0
}

// separation calculates a force to steer away from nearby boids
func separation(index int) (float64, float64) {
	boid := &simulation.Boids[index]
	var steerX, steerY float64

	for i, other := range simulation.Boids {
		if i == index {
			continue
		}

		// Calculate distance with wrapping at screen edges
		dx := math.Abs(boid.X - other.X)
		if dx > simulation.Width/2 {
			dx = simulation.Width - dx
		}

		dy := math.Abs(boid.Y - other.Y)
		if dy > simulation.Height/2 {
			dy = simulation.Height - dy
		}

		dist := math.Sqrt(dx*dx + dy*dy)

		// If too close, add repulsion force
		if dist > 0 && dist <= simulation.params.PerceptionRadius {
			// Calculate direction from other to this boid
			repulsionX := boid.X - other.X
			repulsionY := boid.Y - other.Y

			// Handle wrapping - get shortest direction
			if math.Abs(repulsionX) > simulation.Width/2 {
				if repulsionX > 0 {
					repulsionX -= simulation.Width
				} else {
					repulsionX += simulation.Width
				}
			}

			if math.Abs(repulsionY) > simulation.Height/2 {
				if repulsionY > 0 {
					repulsionY -= simulation.Height
				} else {
					repulsionY += simulation.Height
				}
			}

			// Closer boids have stronger repulsion (inverse distance)
			repulsionX /= dist * dist
			repulsionY /= dist * dist

			steerX += repulsionX
			steerY += repulsionY
		}
	}

	return steerX, steerY
}
