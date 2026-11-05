package router

import (
	"testing"
	"time"

	pb "github.com/distributed-gpu-inference/platform/pkg/proto/inference"
)

func TestLatencyAwareBalancerSelection(t *testing.T) {
	b1 := NewBackend("10.0.0.1:8001", "gpu-pool-1", "A100", 64, 1)
	b2 := NewBackend("10.0.0.2:8001", "gpu-pool-2", "A100", 64, 1)
	b3 := NewBackend("10.0.0.3:8001", "gpu-pool-3", "H100", 128, 2)

	// Simulate load: b1 has high latency, b2 is idle, b3 has active requests
	b1.UpdateLatency(100 * time.Millisecond)
	b2.UpdateLatency(5 * time.Millisecond)
	b3.UpdateLatency(10 * time.Millisecond)

	balancer := NewLatencyAwareBalancer(0.4, 0.4, 0.2)
	req := &pb.InferenceRequest{ModelName: "resnet50"}

	// b2 has lowest latency and 0 active requests -> should be selected
	selected, err := balancer.SelectBackend(req, []*Backend{b1, b2, b3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if selected.Address != b2.Address {
		t.Fatalf("expected b2 (%s), got %s", b2.Address, selected.Address)
	}

	// Now simulate b2 getting busy
	b2.IncActive()
	b2.IncActive()
	b2.IncActive()
	b2.UpdateLatency(50 * time.Millisecond)

	// b3 has lower latency (10ms) and higher weight (2) -> b3 should be preferred
	selected2, err := balancer.SelectBackend(req, []*Backend{b1, b2, b3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if selected2.Address != b3.Address {
		t.Fatalf("expected b3 (%s), got %s", b3.Address, selected2.Address)
	}
}

func TestLatencyAwareBalancerUnhealthyFiltering(t *testing.T) {
	b1 := NewBackend("10.0.0.1:8001", "gpu-pool-1", "A100", 64, 1)
	b2 := NewBackend("10.0.0.2:8001", "gpu-pool-2", "A100", 64, 1)

	// Mark b1 unhealthy
	b1.SetHealthy(false)

	balancer := NewLatencyAwareBalancer(0.4, 0.4, 0.2)
	req := &pb.InferenceRequest{ModelName: "resnet50"}

	selected, err := balancer.SelectBackend(req, []*Backend{b1, b2})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if selected.Address != b2.Address {
		t.Fatalf("expected b2, got unhealthy b1")
	}

	// Mark b2 unhealthy too
	b2.SetHealthy(false)
	_, errAllUnhealthy := balancer.SelectBackend(req, []*Backend{b1, b2})
	if errAllUnhealthy != ErrNoBackendsAvailable {
		t.Fatalf("expected ErrNoBackendsAvailable, got %v", errAllUnhealthy)
	}
}

func TestLatencyAwareBalancerSaturation(t *testing.T) {
	b1 := NewBackend("10.0.0.1:8001", "gpu-pool-1", "A100", 2, 1)

	// Fill to max concurrency
	b1.IncActive()
	b1.IncActive()

	balancer := NewLatencyAwareBalancer(0.4, 0.4, 0.2)
	req := &pb.InferenceRequest{ModelName: "resnet50"}

	_, err := balancer.SelectBackend(req, []*Backend{b1})
	if err != ErrAllBackendsSaturated {
		t.Fatalf("expected ErrAllBackendsSaturated, got %v", err)
	}

	// Decrement active
	b1.DecActive()
	selected, err := balancer.SelectBackend(req, []*Backend{b1})
	if err != nil {
		t.Fatalf("expected success after decrement, got %v", err)
	}
	if selected.Address != b1.Address {
		t.Fatalf("expected b1, got %s", selected.Address)
	}
}

func TestRoundRobinBalancerCycling(t *testing.T) {
	b1 := NewBackend("10.0.0.1:8001", "pool-1", "A100", 64, 1)
	b2 := NewBackend("10.0.0.2:8001", "pool-2", "A100", 64, 1)

	rr := NewRoundRobinBalancer()
	req := &pb.InferenceRequest{ModelName: "test"}

	s1, _ := rr.SelectBackend(req, []*Backend{b1, b2})
	s2, _ := rr.SelectBackend(req, []*Backend{b1, b2})
	s3, _ := rr.SelectBackend(req, []*Backend{b1, b2})

	// Must alternate
	if s1.Address == s2.Address {
		t.Fatalf("expected different backends on consecutive calls: %s vs %s", s1.Address, s2.Address)
	}
	if s1.Address != s3.Address {
		t.Fatalf("expected cycle to return to first backend: %s vs %s", s1.Address, s3.Address)
	}
}

