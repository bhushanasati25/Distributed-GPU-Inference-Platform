package router

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"time"

	"go.uber.org/zap"
)

// CircuitState represents circuit breaker lifecycle status.
type CircuitState int

const (
	CircuitClosed   CircuitState = 0 // Healthy, passing traffic
	CircuitHalfOpen CircuitState = 1 // Probing after cooldown
	CircuitOpen     CircuitState = 2 // Tripped, rejecting traffic
)

type backendHealthTracker struct {
	consecutiveFailures  int
	consecutiveSuccesses int
	state                CircuitState
	lastStateChange      time.Time
}

// HealthChecker periodically interrogates Triton backend instances.
type HealthChecker struct {
	backends           []*Backend
	interval           time.Duration
	timeout            time.Duration
	unhealthyThreshold int
	healthyThreshold   int
	metrics            *MetricsCollector
	logger             *zap.Logger
	httpClient         *http.Client

	mu       sync.Mutex
	trackers map[string]*backendHealthTracker
	stopCh   chan struct{}
}

// NewHealthChecker creates and initializes health monitoring for registered backends.
func NewHealthChecker(
	backends []*Backend,
	interval, timeout time.Duration,
	unhealthyThreshold, healthyThreshold int,
	metrics *MetricsCollector,
	logger *zap.Logger,
) *HealthChecker {
	trackers := make(map[string]*backendHealthTracker)
	for _, b := range backends {
		trackers[b.Address] = &backendHealthTracker{
			state:           CircuitClosed,
			lastStateChange: time.Now(),
		}
	}

	return &HealthChecker{
		backends:           backends,
		interval:           interval,
		timeout:            timeout,
		unhealthyThreshold: unhealthyThreshold,
		healthyThreshold:   healthyThreshold,
		metrics:            metrics,
		logger:             logger.Named("healthcheck"),
		httpClient: &http.Client{
			Timeout: timeout,
		},
		trackers: trackers,
		stopCh:   make(chan struct{}),
	}
}

// Start begins the health check background polling loop.
func (h *HealthChecker) Start(ctx context.Context) {
	ticker := time.NewTicker(h.interval)
	go func() {
		// Run initial check immediately
		h.checkAll(ctx)

		for {
			select {
			case <-ctx.Done():
				ticker.Stop()
				return
			case <-h.stopCh:
				ticker.Stop()
				return
			case <-ticker.C:
				h.checkAll(ctx)
			}
		}
	}()
}

// Stop shuts down the background checking goroutine.
func (h *HealthChecker) Stop() {
	close(h.stopCh)
}

func (h *HealthChecker) checkAll(ctx context.Context) {
	var wg sync.WaitGroup
	for _, b := range h.backends {
		wg.Add(1)
		go func(backend *Backend) {
			defer wg.Done()
			h.checkBackend(ctx, backend)
		}(b)
	}
	wg.Wait()
}

func (h *HealthChecker) checkBackend(ctx context.Context, backend *Backend) {
	// Probe Triton HTTP ready endpoint /v2/health/ready
	url := fmt.Sprintf("http://%s/v2/health/ready", backend.Address)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	healthy := false

	if err == nil {
		resp, err := h.httpClient.Do(req)
		if err == nil {
			if resp.StatusCode == http.StatusOK {
				healthy = true
			}
			_ = resp.Body.Close()
		}
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	tracker := h.trackers[backend.Address]
	if tracker == nil {
		tracker = &backendHealthTracker{state: CircuitClosed}
		h.trackers[backend.Address] = tracker
	}

	if healthy {
		tracker.consecutiveFailures = 0
		tracker.consecutiveSuccesses++

		if tracker.state != CircuitClosed && tracker.consecutiveSuccesses >= h.healthyThreshold {
			tracker.state = CircuitClosed
			tracker.lastStateChange = time.Now()
			backend.SetHealthy(true)
			h.logger.Info("Backend recovered and circuit closed",
				zap.String("backend", backend.Address),
				zap.String("pool", backend.NodePool),
			)
		} else if tracker.state == CircuitClosed {
			backend.SetHealthy(true)
		}
	} else {
		tracker.consecutiveSuccesses = 0
		tracker.consecutiveFailures++

		if tracker.state == CircuitClosed && tracker.consecutiveFailures >= h.unhealthyThreshold {
			tracker.state = CircuitOpen
			tracker.lastStateChange = time.Now()
			backend.SetHealthy(false)
			h.logger.Warn("Backend unhealthy! Circuit tripped to OPEN",
				zap.String("backend", backend.Address),
				zap.String("pool", backend.NodePool),
				zap.Int("failures", tracker.consecutiveFailures),
			)
		} else if tracker.state == CircuitOpen {
			// Check if cooldown elapsed for half-open probe
			if time.Since(tracker.lastStateChange) > 30*time.Second {
				tracker.state = CircuitHalfOpen
				tracker.lastStateChange = time.Now()
				h.logger.Info("Attempting circuit HALF-OPEN probe",
					zap.String("backend", backend.Address),
				)
			}
		}
	}

	if h.metrics != nil {
		h.metrics.SetBackendHealth(backend.Address, backend.IsHealthy())
	}
}

