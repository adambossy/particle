package main

import (
	"fmt"
	"html/template"
	"log"
	"net/http"
	"sync"
)

// Counter struct to safely handle concurrent access
type Counter struct {
	value int
	mutex sync.Mutex
}

// Increment safely increments the counter
func (c *Counter) Increment() {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.value++
}

// GetValue safely returns the current value
func (c *Counter) GetValue() int {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.value
}

var visitCount = &Counter{}

const htmlTemplate = `
<!DOCTYPE html>
<html>
<head>
    <title>Simple Go Web App</title>
</head>
<body>
    <h1>Welcome to My Simple Web App!</h1>
    <div>
        Button clicked: {{.Count}} times
    </div>
    <form method="get">
        <button type="submit">Click Me!</button>
    </form>
</body>
</html>
`

func handler(w http.ResponseWriter, r *http.Request) {
	// Don't increment counter for favicon requests
	if r.URL.Path != "/favicon.ico" {
		visitCount.Increment()
	}

	// Parse template
	tmpl, err := template.New("page").Parse(htmlTemplate)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Create data structure for template
	data := struct {
		Count int
	}{
		Count: visitCount.GetValue(),
	}

	// Execute template
	err = tmpl.Execute(w, data)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

func main() {
	http.HandleFunc("/", handler)
	port := 8002
	fmt.Printf("Server running at http://localhost:%d\n", port)
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", port), nil))
}