package router

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// MetricsCollector manages all Prometheus instrumentation for the router.
type MetricsCollector struct {
	RequestsTotal           *prometheus.CounterVec
	RequestDurationSeconds  *prometheus.HistogramVec
	ActiveRequests          *prometheus.GaugeVec
	QueueDepth              prometheus.Gauge
	BackendHealthy          *prometheus.GaugeVec
	ErrorsTotal             *prometheus.CounterVec
	CrossNodeLatencySeconds *prometheus.HistogramVec
	GRPCPoolActive          prometheus.Gauge
	GRPCPoolSize            prometheus.Gauge
	ForwardedTotal          *prometheus.CounterVec
}

// NewMetricsCollector registers and initialises metrics with standard Prometheus buckets.
func NewMetricsCollector() *MetricsCollector {
	latencyBuckets := []float64{
		0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
	}

	return &MetricsCollector{
		RequestsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "router_requests_total",
				Help: "Total number of inference requests received by the router.",
			},
			[]string{"model", "status", "backend"},
		),
		RequestDurationSeconds: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "router_request_duration_seconds",
				Help:    "End-to-end request latency as observed by the router.",
				Buckets: latencyBuckets,
			},
			[]string{"model", "backend"},
		),
		ActiveRequests: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "router_active_requests",
				Help: "Number of concurrent in-flight requests per backend.",
			},
			[]string{"backend"},
		),
		QueueDepth: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "router_queue_depth",
				Help: "Current count of queued requests waiting for GPU dispatch.",
			},
		),
		BackendHealthy: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "router_backend_healthy",
				Help: "Health status of each backend (1 = healthy, 0 = unhealthy).",
			},
			[]string{"backend"},
		),
		ErrorsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "router_errors_total",
				Help: "Total count of errors encountered during dispatch and execution.",
			},
			[]string{"model", "error_type"},
		),
		CrossNodeLatencySeconds: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "router_cross_node_latency_seconds",
				Help:    "Cross-node pool routing transit latency.",
				Buckets: latencyBuckets,
			},
			[]string{"source_pool", "dest_pool"},
		),
		GRPCPoolActive: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "router_grpc_pool_active",
				Help: "Active gRPC backend client connections currently checked out.",
			},
		),
		GRPCPoolSize: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "router_grpc_pool_size",
				Help: "Total allocated gRPC client connections in the backend pool.",
			},
		),
		ForwardedTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "router_forwarded_total",
				Help: "Total requests forwarded to specific GPU server backends.",
			},
			[]string{"backend"},
		),
	}
}

// RecordRequest records completion of an inference request with latency.
func (m *MetricsCollector) RecordRequest(model, backend, status string, duration time.Duration) {
	m.RequestsTotal.WithLabelValues(model, status, backend).Inc()
	m.RequestDurationSeconds.WithLabelValues(model, backend).Observe(duration.Seconds())
}

// IncActive increments active requests for a backend.
func (m *MetricsCollector) IncActive(backend string) {
	m.ActiveRequests.WithLabelValues(backend).Inc()
}

// DecActive decrements active requests for a backend.
func (m *MetricsCollector) DecActive(backend string) {
	m.ActiveRequests.WithLabelValues(backend).Dec()
}

// SetBackendHealth updates backend readiness status.
func (m *MetricsCollector) SetBackendHealth(backend string, healthy bool) {
	val := 0.0
	if healthy {
		val = 1.0
	}
	m.BackendHealthy.WithLabelValues(backend).Set(val)
}

// RecordError tracks error frequency by category.
func (m *MetricsCollector) RecordError(model, errType string) {
	m.ErrorsTotal.WithLabelValues(model, errType).Inc()
}

// SetQueueDepth updates current queue size gauge.
func (m *MetricsCollector) SetQueueDepth(depth int) {
	m.QueueDepth.Set(float64(depth))
}

