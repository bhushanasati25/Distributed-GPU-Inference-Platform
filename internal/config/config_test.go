package config

import (
	"os"
	"testing"
)

func TestConfigValidation(t *testing.T) {
	// Valid configuration
	cfg := &Config{
		Server: ServerConfig{
			GRPCPort: 50051,
			HTTPPort: 8080,
		},
		Backends: []BackendConfig{
			{Address: "localhost:8001", NodePool: "gpu-pool-1"},
		},
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("expected valid config, got error: %v", err)
	}

	// Invalid gRPC port
	cfgInvalidPort := &Config{
		Server: ServerConfig{
			GRPCPort: 70000,
			HTTPPort: 8080,
		},
		Backends: []BackendConfig{
			{Address: "localhost:8001"},
		},
	}
	if err := cfgInvalidPort.Validate(); err == nil {
		t.Fatal("expected error for invalid grpc port 70000, got nil")
	}

	// Empty backends
	cfgNoBackends := &Config{
		Server: ServerConfig{
			GRPCPort: 50051,
			HTTPPort: 8080,
		},
		Backends: []BackendConfig{},
	}
	if err := cfgNoBackends.Validate(); err == nil {
		t.Fatal("expected error for empty backends, got nil")
	}

	// Backend missing address
	cfgMissingAddr := &Config{
		Server: ServerConfig{
			GRPCPort: 50051,
			HTTPPort: 8080,
		},
		Backends: []BackendConfig{
			{Address: ""},
		},
	}
	if err := cfgMissingAddr.Validate(); err == nil {
		t.Fatal("expected error for backend missing address, got nil")
	}
}

func TestConfigLoadDefaults(t *testing.T) {
	tempFile, err := os.CreateTemp("", "test_config_*.yaml")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer os.Remove(tempFile.Name())

	yamlContent := `
backends:
  - address: "triton-1:8001"
    node_pool: "gpu-pool-1"
`
	if _, err := tempFile.Write([]byte(yamlContent)); err != nil {
		t.Fatalf("failed to write temp config: %v", err)
	}
	_ = tempFile.Close()

	cfg, err := Load(tempFile.Name())
	if err != nil {
		t.Fatalf("failed to load config: %v", err)
	}

	// Check that defaults were filled
	if cfg.Server.GRPCPort != 50051 {
		t.Fatalf("expected default gRPC port 50051, got %d", cfg.Server.GRPCPort)
	}
	if cfg.LoadBalance.Algorithm != "latency_aware" {
		t.Fatalf("expected default algorithm latency_aware, got %s", cfg.LoadBalance.Algorithm)
	}
	if len(cfg.Backends) != 1 {
		t.Fatalf("expected 1 backend, got %d", len(cfg.Backends))
	}
}

