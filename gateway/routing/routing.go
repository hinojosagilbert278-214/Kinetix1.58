// Package routing ports the Python intent gatekeeper and STEPGATE policy.
package routing

import (
	"regexp"
	"strings"
)

const (
	Public  = "public"
	Private = "private"
	Hybrid  = "hybrid"
	Local   = "local"
	Cloud   = "cloud"
)

var (
	publicTerms  = set("latest", "news", "today", "current", "recent", "price", "web", "standard", "rfc", "2025", "2026")
	privateTerms = set("uploaded", "document", "my", "our", "policy", "manual", "notes", "attached", "internal", "file", "spec")
	wordRe       = regexp.MustCompile(`[a-z0-9]+`)
)

func set(ws ...string) map[string]struct{} {
	m := make(map[string]struct{}, len(ws))
	for _, w := range ws {
		m[w] = struct{}{}
	}
	return m
}

// Classify returns the intent route for a query.
func Classify(q string) string {
	pub, priv := false, false
	for _, t := range wordRe.FindAllString(strings.ToLower(q), -1) {
		if _, ok := publicTerms[t]; ok {
			pub = true
		}
		if _, ok := privateTerms[t]; ok {
			priv = true
		}
	}
	switch {
	case pub && priv:
		return Hybrid
	case priv:
		return Private
	}
	return Public
}

// Gate mirrors kinetix/stepgate.py.
type Gate struct {
	BaseLocal, BaseCloud float64
	Window               int
	AlphaMin             float64
}

func DefaultGate() Gate { return Gate{0.35, 0.65, 2048, 0.3} }

func (g Gate) Thresholds(risk, budget float64) (float64, float64) {
	tl := g.BaseLocal * (1 - 0.5*risk)
	tc := g.BaseCloud * (1 - 0.3*risk) * (0.7 + 0.3*budget)
	if tc < tl+1e-6 {
		tc = tl + 1e-6
	}
	return tl, tc
}

// Decide picks the engine. Uncertainty is a caller-supplied proxy (e.g. header) until a calibrated model exists.
func (g Gate) Decide(uncertainty float64, ctxTokens int, risk, budget float64, multiDoc bool) string {
	_, tc := g.Thresholds(risk, budget)
	if ctxTokens > g.Window || multiDoc || uncertainty > tc {
		return Cloud
	}
	return Local
}

// Decision is the routing result for one request.
type Decision struct{ Route, Engine string }

// Route classifies text and applies the gate. Hybrid routes always escalate (cross-domain reasoning).
func (g Gate) Route(text string, uncertainty, risk, budget float64) Decision {
	r := Classify(text)
	tokens := len(wordRe.FindAllString(strings.ToLower(text), -1))
	return Decision{r, g.Decide(uncertainty, tokens, risk, budget, r == Hybrid)}
}
