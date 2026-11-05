package router

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/distributed-gpu-inference/platform/internal/cache"
	"github.com/distributed-gpu-inference/platform/internal/config"
	"github.com/distributed-gpu-inference/platform/internal/scheduler"
	"github.com/distributed-gpu-inference/platform/internal/tracing"
	pb "github.com/distributed-gpu-inference/platform/pkg/proto/inference"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// InferenceRouter serves incoming gRPC inference calls and distributes them to Triton GPU backends.
type InferenceRouter struct {
	pb.UnimplementedInferenceServiceServer

	cfg           *config.Config
	backends      []*Backend
	balancer      LoadBalancer
	healthChecker *HealthChecker
	scheduler     *scheduler.RequestScheduler
	metrics       *MetricsCollector
	logger        *zap.Logger
	cache         *cache.TensorCache
	tracer        *tracing.Tracer

	connMu    sync.RWMutex
	connPool  map[string]*grpc.ClientConn
	clients   map[string]pb.InferenceServiceClient
}

// NewInferenceRouter initializes the router with config, load balancer, health checks, and connection pools.
func NewInferenceRouter(
	cfg *config.Config,
	metrics *MetricsCollector,
	logger *zap.Logger,
) (*InferenceRouter, error) {
	backends := make([]*Backend, 0, len(cfg.Backends))
	for _, bCfg := range cfg.Backends {
		b := NewBackend(
			bCfg.Address,
			bCfg.NodePool,
			bCfg.GPUType,
			bCfg.MaxConcurrent,
			bCfg.Weight,
		)
		backends = append(backends, b)
	}

	balancer := NewLoadBalancer(
		cfg.LoadBalance.Algorithm,
		cfg.LoadBalance.Alpha,
		cfg.LoadBalance.Beta,
		cfg.LoadBalance.Gamma,
	)

	healthChecker := NewHealthChecker(
		backends,
		cfg.HealthCheck.Interval,
		cfg.HealthCheck.Timeout,
		cfg.HealthCheck.UnhealthyThreshold,
		cfg.HealthCheck.HealthyThreshold,
		metrics,
		logger,
	)

	sched := scheduler.NewRequestScheduler(
		cfg.Queue.MaxDepth,
		cfg.Queue.DefaultTimeout,
		logger,
	)

	r := &InferenceRouter{
		cfg:           cfg,
		backends:      backends,
		balancer:      balancer,
		healthChecker: healthChecker,
		scheduler:     sched,
		metrics:       metrics,
		logger:        logger.Named("router"),
		cache:         cache.NewTensorCache(60 * time.Second),
		tracer:        tracing.NewTracer("inference-router", "http://jaeger-service.monitoring:4317", logger),
		connPool:      make(map[string]*grpc.ClientConn),
		clients:       make(map[string]pb.InferenceServiceClient),
	}

	// Link scheduler dispatcher to router execution
	sched.SetDispatcher(r.dispatchDirect)

	return r, nil
}

// Start boots background health checks and request scheduler workers.
func (r *InferenceRouter) Start(ctx context.Context) {
	r.healthChecker.Start(ctx)
	r.scheduler.Start(ctx, 16) // 16 dispatch workers
	r.logger.Info("InferenceRouter background routines active",
		zap.Int("backends", len(r.backends)),
		zap.String("algorithm", r.cfg.LoadBalance.Algorithm),
	)
}

// Close gracefully closes all backend gRPC client connections.
func (r *InferenceRouter) Close() {
	r.healthChecker.Stop()
	r.scheduler.Stop()

	r.connMu.Lock()
	defer r.connMu.Unlock()
	for addr, conn := range r.connPool {
		if err := conn.Close(); err != nil {
			r.logger.Warn("Error closing connection", zap.String("backend", addr), zap.Error(err))
		}
	}
}

// GetBackends returns all registered backends for telemetry and chaos injection.
func (r *InferenceRouter) GetBackends() []*Backend {
	return r.backends
}

// Predict implements pb.InferenceServiceServer.
func (r *InferenceRouter) Predict(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	ctx, span := r.tracer.StartSpan(ctx, "router.predict")
	defer span.End()
	span.SetTag("model.name", req.ModelName)

	startTime := time.Now()

	// 1. Basic validation
	if req.ModelName == "" {
		r.metrics.RecordError("unknown", "invalid_argument")
		return nil, status.Errorf(codes.InvalidArgument, "model_name must not be empty")
	}
	if len(req.Inputs) == 0 {
		r.metrics.RecordError(req.ModelName, "invalid_argument")
		return nil, status.Errorf(codes.InvalidArgument, "inputs tensor list cannot be empty")
	}

	// 2. Check Semantic/Tensor Cache
	if cached, hit := r.cache.Get(req); hit {
		span.SetTag("cache.hit", "true")
		r.metrics.RecordRequest(req.ModelName, "cache", "hit", time.Since(startTime))
		return cached, nil
	}
	span.SetTag("cache.hit", "false")

	// 2. Queue depth metric tracking
	r.metrics.SetQueueDepth(r.scheduler.Depth())

	// 3. Select backend or enqueue under peak load
	backend, err := r.balancer.SelectBackend(req, r.backends)
	if err != nil {
		if err == ErrAllBackendsSaturated {
			// Saturated backends -> enqueue request through priority scheduler
			r.logger.Debug("All backends saturated, enqueuing request",
				zap.String("model", req.ModelName),
				zap.Int32("priority", req.Priority),
			)
			resp, qErr := r.scheduler.Enqueue(ctx, req)
			if qErr != nil {
				r.metrics.RecordError(req.ModelName, "queue_rejected")
				return nil, status.Errorf(codes.ResourceExhausted, "request queue rejected: %v", qErr)
			}
			return resp, nil
		}
		r.metrics.RecordError(req.ModelName, "no_backends")
		return nil, status.Errorf(codes.Unavailable, "inference service unavailable: %v", err)
	}

	// 4. Dispatch directly to selected backend
	resp, dispatchErr := r.dispatchToBackend(ctx, req, backend)
	duration := time.Since(startTime)

	statusStr := "success"
	if dispatchErr != nil {
		statusStr = "error"
		r.metrics.RecordError(req.ModelName, "dispatch_failed")
	} else if resp != nil {
		r.cache.Set(req, resp)
	}
	r.metrics.RecordRequest(req.ModelName, backend.Address, statusStr, duration)

	return resp, dispatchErr
}

func (r *InferenceRouter) dispatchDirect(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	backend, err := r.balancer.SelectBackend(req, r.backends)
	if err != nil {
		return nil, err
	}
	return r.dispatchToBackend(ctx, req, backend)
}

func (r *InferenceRouter) dispatchToBackend(
	ctx context.Context,
	req *pb.InferenceRequest,
	backend *Backend,
) (*pb.InferenceResponse, error) {
	backend.IncActive()
	r.metrics.IncActive(backend.Address)
	r.metrics.ForwardedTotal.WithLabelValues(backend.Address).Inc()

	defer func() {
		backend.DecActive()
		r.metrics.DecActive(backend.Address)
	}()

	client, err := r.getOrCreateClient(backend.Address)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to connect to backend %s: %v", backend.Address, err)
	}

	// Context propagation with deadline
	callCtx := ctx
	if req.DeadlineMs > 0 {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, time.Duration(req.DeadlineMs)*time.Millisecond)
		defer cancel()
	}

	start := time.Now()
	resp, err := client.Predict(callCtx, req)
	elapsed := time.Since(start)

	if err != nil {
		r.logger.Error("Backend execution error",
			zap.String("backend", backend.Address),
			zap.String("model", req.ModelName),
			zap.Error(err),
		)
		return nil, err
	}

	// Update backend moving average latency
	backend.UpdateLatency(elapsed)

	if resp.LatencyMs == 0 {
		resp.LatencyMs = float32(elapsed.Microseconds()) / 1000.0
	}

	return resp, nil
}

func (r *InferenceRouter) getOrCreateClient(address string) (pb.InferenceServiceClient, error) {
	r.connMu.RLock()
	client, exists := r.clients[address]
	r.connMu.RUnlock()
	if exists {
		return client, nil
	}

	r.connMu.Lock()
	defer r.connMu.Unlock()

	// Double-check under write lock
	if client, exists := r.clients[address]; exists {
		return client, nil
	}

	// Dial gRPC connection
	conn, err := grpc.Dial(
		address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(64*1024*1024), // 64MB tensor responses
			grpc.MaxCallSendMsgSize(64*1024*1024), // 64MB tensor inputs
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to dial backend %s: %w", address, err)
	}

	client = pb.NewInferenceServiceClient(conn)
	r.connPool[address] = conn
	r.clients[address] = client

	r.metrics.GRPCPoolSize.Set(float64(len(r.connPool)))

	return client, nil
}

// ModelStatus implements pb.InferenceServiceServer.
func (r *InferenceRouter) ModelStatus(ctx context.Context, req *pb.ModelStatusRequest) (*pb.ModelStatusResponse, error) {
	var backendsStatus []*pb.BackendModelStatus
	isReadyAny := false

	for _, b := range r.backends {
		st := pb.ModelState_MODEL_STATE_READY
		reason := "Backend online"
		if !b.IsHealthy() {
			st = pb.ModelState_MODEL_STATE_UNAVAILABLE
			reason = "Backend health check failed"
		} else {
			isReadyAny = true
		}

		backendsStatus = append(backendsStatus, &pb.BackendModelStatus{
			BackendAddress: b.Address,
			State:          st,
			Reason:         reason,
		})
	}

	overallState := pb.ModelState_MODEL_STATE_UNAVAILABLE
	if isReadyAny {
		overallState = pb.ModelState_MODEL_STATE_READY
	}

	return &pb.ModelStatusResponse{
		ModelName:    req.ModelName,
		ModelVersion: req.ModelVersion,
		State:        overallState,
		Backends:     backendsStatus,
	}, nil
}

// Health implements pb.InferenceServiceServer.
func (r *InferenceRouter) Health(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
	healthyCount := int32(0)
	for _, b := range r.backends {
		if b.IsHealthy() {
			healthyCount++
		}
	}

	status := pb.ServingStatus_SERVING
	msg := fmt.Sprintf("%d of %d backends healthy", healthyCount, len(r.backends))
	if healthyCount == 0 {
		status = pb.ServingStatus_NOT_SERVING
		msg = "All backends unhealthy"
	}

	return &pb.HealthResponse{
		Status:          status,
		HealthyBackends: healthyCount,
		TotalBackends:   int32(len(r.backends)),
		Message:         msg,
	}, nil
}

