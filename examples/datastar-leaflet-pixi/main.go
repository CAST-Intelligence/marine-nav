package main

import (
	"embed"
	_ "embed"
	"fmt"
	"io/fs"
	"log"
	"math"
	"net/http"
	"strconv"
	"sync"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	datastar "github.com/starfederation/datastar/sdk/go"
)

//go:embed index.html
var indexHTML []byte

//go:embed static
var staticFiles embed.FS

// GeographicPoint represents a lat/lng coordinate
type GeographicPoint struct {
	Lat float64 `json:"lat"`
	Lng float64 `json:"lng"`
}

// Polygon represents a set of connected points
type Polygon struct {
	ID         string            `json:"id"`
	Points     []GeographicPoint `json:"points"`
	Color      string            `json:"color"`
	FillColor  string            `json:"fillColor"`
	Properties map[string]string `json:"properties"`
}

// Selection represents a user selection on the map
type Selection struct {
	Center   GeographicPoint `json:"center"`
	Distance float64         `json:"distance"`
}

// AppState holds the current state of the application
type AppState struct {
	Polygons     []Polygon
	UserSelection *Selection
	mu           sync.RWMutex
}

// SelectionRequest is used for reading selection changes from client
type SelectionRequest struct {
	Center   *GeographicPoint `json:"center"`
	Distance *float64         `json:"distance"`
}

// Global app state
var appState AppState

func init() {
	// Initialize with empty data
	appState = AppState{
		Polygons:     []Polygon{},
		UserSelection: nil,
	}
}

func main() {
	// Setup router
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)

	// Serve the main HTML page
	r.Get("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write(indexHTML)
	})

	// Endpoint to stream polygon data
	r.Get("/map-stream", streamMapData)

	// Endpoint to update user selection
	r.Get("/update-selection", updateSelection)

	// Endpoint to create polygon
	r.Get("/create-polygon", createPolygon)

	// Serve static files
	staticFS, err := fs.Sub(staticFiles, "static")
	if err != nil {
		log.Fatalf("Failed to create sub filesystem: %v", err)
	}
	r.Handle("/static/*", http.StripPrefix("/static/", http.FileServer(http.FS(staticFS))))

	// Start server
	port := ":8080"
	log.Printf("Map visualization server starting on http://localhost%s", port)
	if err := http.ListenAndServe(port, r); err != nil {
		log.Fatalf("Error starting server: %v", err)
	}
}

// streamMapData streams the current state to clients using SSE
func streamMapData(w http.ResponseWriter, r *http.Request) {
	sse := datastar.NewSSE(w, r)

	// Send initial state
	appState.mu.RLock()
	err := sse.MarshalAndMergeSignals(map[string]interface{}{
		"polygons":      appState.Polygons,
		"userSelection": appState.UserSelection,
	})
	appState.mu.RUnlock()

	if err != nil {
		log.Printf("Error sending initial map state: %v", err)
		return
	}

	// Wait for this connection to close
	<-r.Context().Done()
	log.Println("Client disconnected from map stream")
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

// updateSelection updates the user's selected point and radius
func updateSelection(w http.ResponseWriter, r *http.Request) {
	// Use struct to receive signals directly
	reqData := &SelectionRequest{}

	// Read signals from request directly into struct
	if err := datastar.ReadSignals(r, reqData); err != nil {
		log.Printf("Error reading signals for selection: %v", err)
		return
	}

	// Update state
	appState.mu.Lock()
	if reqData.Center != nil && reqData.Distance != nil {
		appState.UserSelection = &Selection{
			Center:   *reqData.Center,
			Distance: *reqData.Distance,
		}
	}
	appState.mu.Unlock()

	// Send updated state to all clients
	sse := datastar.NewSSE(w, r)
	appState.mu.RLock()
	err := sse.MarshalAndMergeSignals(map[string]interface{}{
		"userSelection": appState.UserSelection,
	})
	appState.mu.RUnlock()

	if err != nil {
		log.Printf("Error sending updated selection: %v", err)
	}
}

// PolygonRequest is used for reading polygon creation parameters from client
type PolygonRequest struct {
	PolygonID  string `json:"polygonID"`
	Timestamp  int64  `json:"timestamp"`
	Color      string `json:"color"`
	FillColor  string `json:"fillColor"`
}

// createPolygon creates a new polygon around the current selection
func createPolygon(w http.ResponseWriter, r *http.Request) {
	// Use struct to receive signals directly
	reqData := &PolygonRequest{}

	// Read signals from request directly into struct
	if err := datastar.ReadSignals(r, reqData); err != nil {
		log.Printf("Error reading signals for polygon creation: %v", err)
		return
	}

	// Check if we have the required parameter
	hasID := reqData.PolygonID != ""
	polygonID := reqData.PolygonID

	appState.mu.Lock()
	// Make sure we have a selection
	if appState.UserSelection == nil {
		appState.mu.Unlock()
		log.Printf("Cannot create polygon: No active selection")
		http.Error(w, "No active selection", http.StatusBadRequest)
		return
	}

	// Create polygon from the current selection
	if hasID {
		center := appState.UserSelection.Center
		radius := appState.UserSelection.Distance

		// Generate points in a circle around the center
		numPoints := 12 // Number of points to generate
		points := make([]GeographicPoint, numPoints)

		for i := 0; i < numPoints; i++ {
			// Calculate point at angle around the center
			angle := float64(i) * (2 * math.Pi / float64(numPoints))
			
			// Convert meters to approximate decimal degrees
			// This is a simple approximation for small distances
			latRadians := center.Lat * math.Pi / 180
			
			// Distance in degrees lat/lng 
			// (very rough approximation; for accuracy use proper Haversine formulas)
			latOffset := radius / 111000 * math.Sin(angle)          // 1 degree lat is ~111km
			lngOffset := radius / (111000 * math.Cos(latRadians)) * math.Cos(angle)
			
			points[i] = GeographicPoint{
				Lat: center.Lat + latOffset,
				Lng: center.Lng + lngOffset,
			}
		}

		// Add a new polygon
		newPolygon := Polygon{
			ID:        polygonID,
			Points:    points,
			Color:     "#FF0000", // Default red outline
			FillColor: "#FF000055", // Default semi-transparent red fill
			Properties: map[string]string{
				"createdAt": fmt.Sprintf("%d", reqData.Timestamp),
				"name":      fmt.Sprintf("Polygon %s", polygonID),
			},
		}

		// If custom color provided, use it
		if reqData.Color != "" {
			newPolygon.Color = reqData.Color
		}
		if reqData.FillColor != "" {
			newPolygon.FillColor = reqData.FillColor
		}

		// Add the new polygon to our state
		appState.Polygons = append(appState.Polygons, newPolygon)
	}
	appState.mu.Unlock()

	// Send updated polygons to all clients
	sse := datastar.NewSSE(w, r)
	appState.mu.RLock()
	err := sse.MarshalAndMergeSignals(map[string]interface{}{
		"polygons": appState.Polygons,
	})
	appState.mu.RUnlock()

	if err != nil {
		log.Printf("Error sending updated polygons: %v", err)
	}
}