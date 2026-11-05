package tracing

import (
	"context"
	"fmt"
	"time"

	"go.uber.org/zap"
)

// Span represents a lightweight distributed trace span record.
type Span struct {
	TraceID   string
	SpanID    string
	Name      string
	StartTime time.Time
	Duration  time.Duration
	Tags      map[string]string
}

// Tracer manages trace context generation and export to OTLP collectors.
type Tracer struct {
	serviceName  string
	collectorURL string
	logger       *zap.Logger
}

// NewTracer creates an OpenTelemetry-compatible tracing helper.
func NewTracer(serviceName, collectorURL string, logger *zap.Logger) *Tracer {
	return &Tracer{
		serviceName:  serviceName,
		collectorURL: collectorURL,
		logger:       logger.Named("tracer"),
	}
}

// StartSpan creates a new timed span context.
func (t *Tracer) StartSpan(ctx context.Context, spanName string) (context.Context, *Span) {
	s := &Span{
		TraceID:   fmt.Sprintf("trace-%d", time.Now().UnixNano()),
		SpanID:    fmt.Sprintf("span-%d", time.Now().UnixNano()),
		Name:      spanName,
		StartTime: time.Now(),
		Tags:      make(map[string]string),
	}
	s.Tags["service.name"] = t.serviceName
	return ctx, s
}

// SetTag adds key-value metadata to the span.
func (s *Span) SetTag(key, value string) {
	s.Tags[key] = value
}

// End finishes span recording and computes execution duration.
func (s *Span) End() {
	s.Duration = time.Since(s.StartTime)
}

