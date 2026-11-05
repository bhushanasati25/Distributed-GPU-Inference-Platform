package scheduler

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	pb "github.com/distributed-gpu-inference/platform/pkg/proto/inference"
	"go.uber.org/zap"
)

var (
	// ErrQueueFull is returned when queue capacity has been exceeded.
	ErrQueueFull = errors.New("inference request queue is full: backpressure active")
	// ErrRequestTimeout is returned when request deadline elapsed in queue.
	ErrRequestTimeout = errors.New("request timed out waiting in priority queue")
)

// DispatcherFunc is called when a queued request is dequeued for execution.
type DispatcherFunc func(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error)

// RequestScheduler coordinates request queuing, backpressure, and dispatch across GPU pools.
type RequestScheduler struct {
	maxDepth       int
	defaultTimeout time.Duration
	queue          *PriorityQueue
	logger         *zap.Logger

	mu         sync.Mutex
	stopCh     chan struct{}
	dispatcher DispatcherFunc
}

// NewRequestScheduler creates a priority scheduler.
func NewRequestScheduler(maxDepth int, defaultTimeout time.Duration, logger *zap.Logger) *RequestScheduler {
	return &RequestScheduler{
		maxDepth:       maxDepth,
		defaultTimeout: defaultTimeout,
		queue:          NewPriorityQueue(),
		logger:         logger.Named("scheduler"),
		stopCh:         make(chan struct{}),
	}
}

// SetDispatcher configures the callback executed to route and serve dequeued requests.
func (s *RequestScheduler) SetDispatcher(fn DispatcherFunc) {
	s.mu.Lock()
	s.dispatcher = fn
	s.mu.Unlock()
}

// Start runs background eviction and dispatch workers.
func (s *RequestScheduler) Start(ctx context.Context, numWorkers int) {
	// Start deadline eviction timer
	go s.evictionLoop(ctx)

	// Start concurrent dequeue dispatchers
	for i := 0; i < numWorkers; i++ {
		go s.workerLoop(ctx, i)
	}
}

// Stop shuts down scheduler loops.
func (s *RequestScheduler) Stop() {
	close(s.stopCh)
}

// Enqueue submits a request to the priority queue. It blocks until executed or times out.
func (s *RequestScheduler) Enqueue(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	if s.queue.Len() >= s.maxDepth {
		return nil, ErrQueueFull
	}

	deadline := time.Now().Add(s.defaultTimeout)
	if req.DeadlineMs > 0 {
		explicitDeadline := time.Now().Add(time.Duration(req.DeadlineMs) * time.Millisecond)
		if explicitDeadline.Before(deadline) {
			deadline = explicitDeadline
		}
	}

	respChan := make(chan QueuedResponse, 1)
	item := &QueueItem{
		Request:      req,
		Priority:     req.Priority,
		EnqueuedAt:   time.Now(),
		Deadline:     deadline,
		ResponseChan: respChan,
	}

	s.queue.Push(item)

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(time.Until(deadline)):
		return nil, ErrRequestTimeout
	case result := <-respChan:
		return result.Response, result.Err
	}
}

// Depth returns the current count of queued items.
func (s *RequestScheduler) Depth() int {
	return s.queue.Len()
}

func (s *RequestScheduler) workerLoop(ctx context.Context, workerID int) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-s.stopCh:
			return
		default:
			item := s.queue.Pop()
			if item == nil {
				time.Sleep(5 * time.Millisecond)
				continue
			}

			// Check if already expired
			if !item.Deadline.IsZero() && time.Now().After(item.Deadline) {
				item.ResponseChan <- QueuedResponse{Err: ErrRequestTimeout}
				close(item.ResponseChan)
				continue
			}

			s.mu.Lock()
			fn := s.dispatcher
			s.mu.Unlock()

			if fn == nil {
				item.ResponseChan <- QueuedResponse{Err: fmt.Errorf("no dispatcher configured")}
				close(item.ResponseChan)
				continue
			}

			// Execute inference
			resp, err := fn(ctx, item.Request)
			item.ResponseChan <- QueuedResponse{Response: resp, Err: err}
			close(item.ResponseChan)
		}
	}
}

func (s *RequestScheduler) evictionLoop(ctx context.Context) {
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-s.stopCh:
			return
		case now := <-ticker.C:
			evicted := s.queue.EvictExpired(now, ErrRequestTimeout)
			if evicted > 0 {
				s.logger.Warn("Evicted expired inference requests from queue",
					zap.Int("count", evicted),
					zap.Int("remaining", s.queue.Len()),
				)
			}
		}
	}
}

