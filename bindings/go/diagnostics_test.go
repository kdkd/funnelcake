package funnelcake

import "testing"

func TestDiagnostics(t *testing.T) {
	if Version() == "" {
		t.Fatal("missing version")
	}
	switch Backend() {
	case "scalar", "avx2", "avx512", "neon", "rvv":
	default:
		t.Fatal("unknown backend")
	}
}
