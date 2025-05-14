package main

import (
	"bytes"
	"embed"
	_ "embed"
	"io/fs"
	"log"
	"math/rand"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	datastar "github.com/starfederation/datastar/sdk/go"
)

//go:embed index.html
var indexHTML []byte

//go:embed static
var staticFiles embed.FS

// --- Struct for reading signals from mangle request ---
type MangleRequest struct {
	Name string `json:"name"` // Must match the signal name ($name)
}

// --- Random characters for mangling ---
const letterBytes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()"

func main() {
	// Seed random number generator (use better seeding in production)
	rand.Seed(time.Now().UnixNano())

	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)

	// --- Routes ---

	// Serve the main HTML page
	r.Get("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write(indexHTML)
	})

	// --- NEW: Endpoint to mangle text ---
	r.Get("/mangle-text", func(w http.ResponseWriter, r *http.Request) {

		reqData := &MangleRequest{}
		if err := datastar.ReadSignals(r, reqData); err != nil {
			log.Printf("Error reading signals for mangle: %v", err)
			// sse.ConsoleError(err) // Inform client console
			return // Don't proceed
		}

		// Perform the "mangling"
		mangled := mangleString(reqData.Name)

		// Prepare the signal to send back
		responseSignal := map[string]string{
			"mangledText": mangled, // Key must match $mangledText in HTML
		}

		// Send the updated signal back via SSE
		sse := datastar.NewSSE(w, r) // Initialize SSE

		err := sse.MarshalAndMergeSignals(responseSignal)
		if err != nil {
			log.Printf("Error merging mangledText signal: %v", err)
			// Connection might be closed, often can't send ConsoleError
		}
	})
	// --- End New Endpoint ---

	// --- Serve Static Files (for reverse-component.js) ---
	staticFS, err := fs.Sub(staticFiles, "static")
	if err != nil {
		log.Fatalf("Failed to create sub filesystem: %v", err)
	}
	r.Handle("/static/*", http.StripPrefix("/static/", http.FileServer(http.FS(staticFS))))

	// --- Start Server ---
	port := ":8080"
	log.Printf("Server starting on http://localhost%s", port)
	if err := http.ListenAndServe(port, r); err != nil {
		log.Fatalf("Error starting server: %v", err)
	}
}

// --- Helper function to mangle the string ---
func mangleString(s string) string {
	if s == "" {
		return ""
	}
	var result bytes.Buffer
	runes := []rune(s) // Work with runes for Unicode safety

	for i, r := range runes {
		result.WriteRune(r)
		// Add 1 or 2 random chars after each original char, except the last
		if i < len(runes)-1 {
			count := 1 + rand.Intn(2) // 1 or 2
			for j := 0; j < count; j++ {
				result.WriteByte(letterBytes[rand.Intn(len(letterBytes))])
			}
		}
	}
	return result.String()
}
