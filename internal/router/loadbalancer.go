package router

import (
	"errors"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	pb "github.com/distributed-gpu-inference/platform/pkg/proto/inference"
)

var (
	// ErrNoBackendsAvailable is returned when no healthy backend is registered.
	ErrNoBackendsAvailable = errors.New("no healthy backends available to serve request")
	// ErrAllBackendsSaturated is returned when all healthy backends exceed max concurrency.
	ErrAllBackendsSaturated = errors.New("all healthy backends are currently saturated")
)

// Backend represents a single Triton GPU server node and its dynamic operational telemetry.
type Backend struct {
	Address       string
	NodePool      string
	GPUType       string
	MaxConcurrent int
	Weight        int

	activeRequests int64
	avgLatencyNs   int64 // Exponential moving average in nanoseconds
	healthy        int32 // 1 = true, 0 = false

	mu              sync.RWMutex
	lastHealthCheck time.Time
}

// NewBackend creates a new Backend descriptor.
func NewBackend(addr, nodePool, gpuType string, maxConcurrent, weight int) *Backend {
	b := &Backend{
		Address:       addr,
		NodePool:      nodePool,
		GPUType:       gpuType,
		MaxConcurrent: maxConcurrent,
		Weight:        weight,
	}
	atomic.StoreInt32(&b.healthy, 1)
	atomic.StoreInt64(&b.avgLatencyNs, int64(10*time.Millisecond))
	return b
}

// ActiveRequests returns current count of in-flight inference requests.
func (b *Backend) ActiveRequests() int64 {
	return atomic.LoadInt64(&b.activeRequests)
}

// IncActive increments active requests atomically.
func (b *Backend) IncActive() int64 {
	return atomic.AddInt64(&b.activeRequests, 1)
}

// DecActive decrements active requests atomically.
func (b *Backend) DecActive() int64 {
	return atomic.AddInt64(&b.activeRequests, -1)
}

// AvgLatency returns current moving average latency.
func (b *Backend) AvgLatency() time.Duration {
	ns := atomic.LoadInt64(&b.avgLatencyNs)
	return time.Duration(ns)
}

// SetLatency sets average latency directly without EMA decay.
func (b *Backend) SetLatency(d time.Duration) {
	atomic.StoreInt64(&b.avgLatencyNs, d.Nanoseconds())
}

// UpdateLatency applies exponential moving average (EMA) smoothing to latency tracking.
func (b *Backend) UpdateLatency(d time.Duration) {
	const decay = 0.2 // Weight given to newest observation
	newNs := d.Nanoseconds()
	for {
		oldNs := atomic.LoadInt64(&b.avgLatencyNs)
		updated := int64(float64(oldNs)*(1.0-decay) + float64(newNs)*decay)
		if atomic.CompareAndSwapInt64(&b.avgLatencyNs, oldNs, updated) {
			break
		}
	}
}

// IsHealthy returns true if backend passed latest health probes.
func (b *Backend) IsHealthy() bool {
	return atomic.LoadInt32(&b.healthy) == 1
}

// SetHealthy updates backend health status.
func (b *Backend) SetHealthy(ok bool) {
	val := int32(0)
	if ok {
		val = 1
	}
	atomic.StoreInt32(&b.healthy, val)
	b.mu.Lock()
	b.lastHealthCheck = time.Now()
	b.mu.Unlock()
}

// LoadBalancer selects an optimal backend for an incoming inference request.
type LoadBalancer interface {
	SelectBackend(req *pb.InferenceRequest, backends []*Backend) (*Backend, error)
}

// LatencyAwareBalancer selects nodes based on in-flight load, measured latency, and data locality.
type LatencyAwareBalancer struct {
	Alpha float64 // Active requests weight
	Beta  float64 // Moving average latency weight
	Gamma float64 // Queue depth / capacity weight
}

// NewLatencyAwareBalancer creates an instance with specified tuning weights.
func NewLatencyAwareBalancer(alpha, beta, gamma float64) *LatencyAwareBalancer {
	return &LatencyAwareBalancer{
		Alpha: alpha,
		Beta:  beta,
		Gamma: gamma,
	}
}

// SelectBackend evaluates all healthy backends and returns the one with lowest composite score.
func (lb *LatencyAwareBalancer) SelectBackend(req *pb.InferenceRequest, backends []*Backend) (*Backend, error) {
	var best *Backend
	lowestScore := 1e18

	// Gather healthy backends
	healthy := make([]*Backend, 0, len(backends))
	for _, b := range backends {
		if b.IsHealthy() {
			healthy = append(healthy, b)
		}
	}

	if len(healthy) == 0 {
		return nil, ErrNoBackendsAvailable
	}

	for _, b := range healthy {
		active := b.ActiveRequests()
		if b.MaxConcurrent > 0 && int(active) >= b.MaxConcurrent {
			continue // Saturated backend
		}

		// Calculate composite score
		latencyMs := float64(b.AvgLatency().Microseconds()) / 1000.0
		capacityRatio := float64(active)
		if b.MaxConcurrent > 0 {
			capacityRatio = float64(active) / float64(b.MaxConcurrent)
		}

		score := (lb.Alpha * float64(active)) +
			(lb.Beta * latencyMs) +
			(lb.Gamma * capacityRatio * 10.0)

		// Weight adjustment (higher weight lowers score)
		if b.Weight > 1 {
			score = score / float64(b.Weight)
		}

		if score < lowestScore {
			lowestScore = score
			best = b
		}
	}

	if best == nil {
		return nil, ErrAllBackendsSaturated
	}

	return best, nil
}

// RoundRobinBalancer implements simple cycling across healthy backends.
type RoundRobinBalancer struct {
	idx uint64
}

// NewRoundRobinBalancer creates a round robin balancer.
func NewRoundRobinBalancer() *RoundRobinBalancer {
	return &RoundRobinBalancer{}
}

// SelectBackend selects the next healthy backend in sequence.
func (rr *RoundRobinBalancer) SelectBackend(req *pb.InferenceRequest, backends []*Backend) (*Backend, error) {
	healthy := make([]*Backend, 0, len(backends))
	for _, b := range backends {
		if b.IsHealthy() {
			healthy = append(healthy, b)
		}
	}
	if len(healthy) == 0 {
		return nil, ErrNoBackendsAvailable
	}

	n := atomic.AddUint64(&rr.idx, 1)
	return healthy[n%uint64(len(healthy))], nil
}

// WeightedRandomBalancer picks randomly proportional to assigned backend weights.
type WeightedRandomBalancer struct {
	rng *rand.Rand
	mu  sync.Mutex
}

// NewWeightedRandomBalancer creates a weighted random balancer.
func NewWeightedRandomBalancer() *WeightedRandomBalancer {
	return &WeightedRandomBalancer{
		rng: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// SelectBackend selects a backend weighted by its configuration weight.
func (w *WeightedRandomBalancer) SelectBackend(req *pb.InferenceRequest, backends []*Backend) (*Backend, error) {
	healthy := make([]*Backend, 0, len(backends))
	totalWeight := 0
	for _, b := range backends {
		if b.IsHealthy() {
			weight := b.Weight
			if weight <= 0 {
				weight = 1
			}
			healthy = append(healthy, b)
			totalWeight += weight
		}
	}
	if len(healthy) == 0 {
		return nil, ErrNoBackendsAvailable
	}

	w.mu.Lock()
	r := w.rng.Intn(totalWeight)
	w.mu.Unlock()

	running := 0
	for _, b := range healthy {
		weight := b.Weight
		if weight <= 0 {
			weight = 1
		}
		running += weight
		if r < running {
			return b, nil
		}
	}
	return healthy[0], nil
}

// NewLoadBalancer factory constructs chosen algorithm.
func NewLoadBalancer(algo string, alpha, beta, gamma float64) LoadBalancer {
	switch algo {
	case "round_robin":
		return NewRoundRobinBalancer()
	case "weighted_random":
		return NewWeightedRandomBalancer()
	case "latency_aware":
		fallthrough
	default:
		return NewLatencyAwareBalancer(alpha, beta, gamma)
	}
}

