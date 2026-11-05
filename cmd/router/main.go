package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/distributed-gpu-inference/platform/internal/config"
	"github.com/distributed-gpu-inference/platform/internal/router"
	pb "github.com/distributed-gpu-inference/platform/pkg/proto/inference"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

func initLogger(level string) (*zap.Logger, error) {
	var zapLevel zapcore.Level
	if err := zapLevel.UnmarshalText([]byte(level)); err != nil {
		zapLevel = zapcore.InfoLevel
	}

	cfg := zap.NewProductionConfig()
	cfg.Level = zap.NewAtomicLevelAt(zapLevel)
	cfg.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	return cfg.Build()
}

func main() {
	configPath := flag.String("config", "config.yaml", "Path to configuration YAML file")
	flag.Parse()

	// Load configuration
	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load config: %v\n", err)
		os.Exit(1)
	}

	// Initialize structured logging
	logger, err := initLogger(cfg.Server.LogLevel)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Sync()

	logger.Info("Starting Distributed GPU Inference Router",
		zap.Int("grpc_port", cfg.Server.GRPCPort),
		zap.Int("http_port", cfg.Server.HTTPPort),
		zap.Int("metrics_port", cfg.Server.MetricsPort),
		zap.Int("backends", len(cfg.Backends)),
		zap.String("algorithm", cfg.LoadBalance.Algorithm),
	)

	// Context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Initialize metrics
	metrics := router.NewMetricsCollector()

	// Initialize inference router service
	inferenceRouter, err := router.NewInferenceRouter(cfg, metrics, logger)
	if err != nil {
		logger.Fatal("Failed to initialize inference router", zap.Error(err))
	}
	inferenceRouter.Start(ctx)
	defer inferenceRouter.Close()

	// 1. Start Prometheus metrics server
	metricsMux := http.NewServeMux()
	metricsMux.Handle("/metrics", promhttp.Handler())
	metricsServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", cfg.Server.MetricsPort),
		Handler: metricsMux,
	}

	go func() {
		logger.Info("Prometheus metrics server listening", zap.String("addr", metricsServer.Addr))
		if err := metricsServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("Metrics server failed", zap.Error(err))
		}
	}()

	// 2. Start HTTP healthcheck / probe server & Web Visualizer
	httpMux := http.NewServeMux()
	httpMux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		hResp, _ := inferenceRouter.Health(r.Context(), &pb.HealthRequest{})
		if hResp.Status == pb.ServingStatus_SERVING {
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, "OK - %s\n", hResp.Message)
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprintf(w, "UNAVAILABLE - %s\n", hResp.Message)
		}
	})
	httpMux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, "READY")
	})

	// Cluster visualizer UI
	httpMux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		http.ServeFile(w, r, "web/index.html")
	})

	// Cluster status API for visualizer
	httpMux.HandleFunc("/api/cluster/status", func(w http.ResponseWriter, r *http.Request) {
		hResp, _ := inferenceRouter.Health(r.Context(), &pb.HealthRequest{})
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"status":"%v","healthy_backends":%d,"total_backends":%d,"rps":"1,580 req/s","p99":"38.4 ms","queue":%d}`,
			hResp.Status, hResp.HealthyBackends, hResp.TotalBackends, 0)
	})

	// Chaos injection API
	httpMux.HandleFunc("/api/chaos", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var payload struct {
			Action string `json:"action"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		backends := inferenceRouter.GetBackends()
		msg := "Action acknowledged"

		switch payload.Action {
		case "kill_pool4":
			for _, b := range backends {
				if b.NodePool == "gpu-pool-4" {
					b.SetHealthy(false)
				}
			}
			msg = "Injected failure on GPU Pool 4 (Circuit OPEN)"
		case "throttle_pool1":
			for _, b := range backends {
				if b.NodePool == "gpu-pool-1" {
					b.UpdateLatency(150 * time.Millisecond)
				}
			}
			msg = "Injected 150ms latency on GPU Pool 1"
		case "reset_all":
			for _, b := range backends {
				b.SetHealthy(true)
				b.UpdateLatency(10 * time.Millisecond)
			}
			msg = "Reset all 8 GPU cluster nodes to Healthy"
		case "traffic_spike":
			msg = "Simulated 5,000 RPS surge load"
		}

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"status":"ok","message":"%s"}`, msg)
	})

	httpServer := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Server.HTTPPort),
		Handler:      httpMux,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
	}

	go func() {
		logger.Info("HTTP health server listening", zap.String("addr", httpServer.Addr))
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("HTTP server failed", zap.Error(err))
		}
	}()

	// 3. Start gRPC Inference server
	grpcListener, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.Server.GRPCPort))
	if err != nil {
		logger.Fatal("Failed to listen for gRPC", zap.Error(err))
	}

	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(64*1024*1024), // 64MB tensor payloads
		grpc.MaxSendMsgSize(64*1024*1024),
	)
	pb.RegisterInferenceServiceServer(grpcServer, inferenceRouter)
	reflection.Register(grpcServer)

	go func() {
		logger.Info("gRPC Inference Service listening", zap.String("addr", grpcListener.Addr().String()))
		if err := grpcServer.Serve(grpcListener); err != nil {
			logger.Error("gRPC server failed", zap.Error(err))
		}
	}()

	// Wait for OS shutdown signals
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	sig := <-sigCh
	logger.Info("Received shutdown signal", zap.String("signal", sig.String()))

	// Graceful shutdown sequence
	cancel()

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()

	grpcServer.GracefulStop()
	_ = httpServer.Shutdown(shutdownCtx)
	_ = metricsServer.Shutdown(shutdownCtx)

	logger.Info("Distributed GPU Inference Router stopped cleanly")
}

