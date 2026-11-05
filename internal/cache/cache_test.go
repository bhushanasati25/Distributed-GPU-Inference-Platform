package cache

import (
	"testing"
	"time"

	pb "github.com/distributed-gpu-inference/platform/pkg/proto/inference"
)

func TestTensorCacheHitAndMiss(t *testing.T) {
	c := NewTensorCache(1 * time.Second)

	req1 := &pb.InferenceRequest{
		ModelName:    "resnet50",
		ModelVersion: "1",
		Inputs: []*pb.TensorData{
			{Name: "input", Datatype: "FP32", RawData: []byte{1, 2, 3, 4}},
		},
	}

	// Initial miss
	_, ok := c.Get(req1)
	if ok {
		t.Fatal("expected cache miss, got hit")
	}

	// Store response
	resp := &pb.InferenceResponse{
		ModelName: "resnet50",
		LatencyMs: 12.5,
	}
	c.Set(req1, resp)

	// Second query should hit
	cached, ok := c.Get(req1)
	if !ok {
		t.Fatal("expected cache hit, got miss")
	}
	if cached.LatencyMs != 12.5 {
		t.Fatalf("expected latency 12.5, got %f", cached.LatencyMs)
	}

	hits, misses, ratio := c.Stats()
	if hits != 1 || misses != 1 {
		t.Fatalf("expected 1 hit, 1 miss, got %d hits, %d misses", hits, misses)
	}
	if ratio != 50.0 {
		t.Fatalf("expected 50%% ratio, got %f", ratio)
	}
}

func TestTensorCacheTTLExpiration(t *testing.T) {
	c := NewTensorCache(50 * time.Millisecond)

	req := &pb.InferenceRequest{
		ModelName: "test",
		Inputs: []*pb.TensorData{
			{Name: "data", RawData: []byte{0xAA, 0xBB}},
		},
	}
	c.Set(req, &pb.InferenceResponse{ModelName: "test"})

	// Immediate get hits
	if _, ok := c.Get(req); !ok {
		t.Fatal("expected immediate hit")
	}

	// Sleep past TTL
	time.Sleep(70 * time.Millisecond)
	if _, ok := c.Get(req); ok {
		t.Fatal("expected expired miss after TTL, got hit")
	}
}

