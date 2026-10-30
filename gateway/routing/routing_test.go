package routing

import "testing"

func TestClassify(t *testing.T) {
	cases := map[string]string{
		"latest news on rust":                         Public,
		"summarize my uploaded policy":                Private,
		"compare our internal spec to the latest RFC": Hybrid,
	}
	for q, want := range cases {
		if got := Classify(q); got != want {
			t.Errorf("%q: got %s want %s", q, got, want)
		}
	}
}

func TestGate(t *testing.T) {
	g := DefaultGate()
	if g.Route("summarize my policy", 0.1, 0, 1).Engine != Local {
		t.Error("expected local")
	}
	if g.Route("summarize my policy", 0.9, 0, 1).Engine != Cloud {
		t.Error("expected cloud on high uncertainty")
	}
	if g.Route("compare our spec with the latest rfc", 0.1, 0, 1).Engine != Cloud {
		t.Error("expected cloud for hybrid")
	}
	tl, tc := g.Thresholds(1, 0)
	if tl >= tc {
		t.Error("tau_L must be < tau_C")
	}
}
