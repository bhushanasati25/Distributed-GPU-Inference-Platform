package tracing

import (
	"context"
	"testing"
	"time"

	"go.uber.org/zap"
)

func TestTracerSpanLifecycle(t *testing.T) {
	logger := zap.NewNop()
	tracer := NewTracer("test-service", "http://localhost:4317", logger)

	ctx := context.Background()
	_, span := tracer.StartSpan(ctx, "test.inference.predict")

	if span.Name != "test.inference.predict" {
		t.Fatalf("expected span name 'test.inference.predict', got %s", span.Name)
	}
	if span.Tags["service.name"] != "test-service" {
		t.Fatalf("expected service tag 'test-service', got %s", span.Tags["service.name"])
	}

	span.SetTag("model.name", "resnet50")
	span.SetTag("backend.pool", "gpu-pool-3")

	time.Sleep(10 * time.Millisecond)
	span.End()

	if span.Duration < 10*time.Millisecond {
		t.Fatalf("expected duration >= 10ms, got %v", span.Duration)
	}
	if span.Tags["model.name"] != "resnet50" {
		t.Fatalf("expected tag model.name resnet50, got %s", span.Tags["model.name"])
	}
}

