// Command mock is a stand-in LLM upstream that reports which engine served the request.
package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
)

func main() {
	name := os.Getenv("NAME")
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `{"served_by":%q,"route":%q,"engine":%q}`+"\n", name, r.Header.Get("x-kinetix-route"), r.Header.Get("x-kinetix-engine"))
	})
	log.Fatal(http.ListenAndServe(":8000", nil))
}
