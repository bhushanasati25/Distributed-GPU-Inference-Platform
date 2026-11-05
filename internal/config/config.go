package config

import (
	"fmt"
	"strings"
	"time"

	"github.com/spf13/viper"
)

// Config aggregates all operational parameters for the inference router.
type Config struct {
	Server      ServerConfig      `mapstructure:"server"`
	Backends    []BackendConfig   `mapstructure:"backends"`
	LoadBalance LoadBalanceConfig `mapstructure:"load_balancing"`
	Queue       QueueConfig       `mapstructure:"queue"`
	Redis       RedisConfig       `mapstructure:"redis"`
	HealthCheck HealthCheckConfig `mapstructure:"health_check"`
}

// ServerConfig configures listener ports and timeouts.
type ServerConfig struct {
	GRPCPort     int           `mapstructure:"grpc_port"`
	HTTPPort     int           `mapstructure:"http_port"`
	MetricsPort  int           `mapstructure:"metrics_port"`
	ReadTimeout  time.Duration `mapstructure:"read_timeout"`
	WriteTimeout time.Duration `mapstructure:"write_timeout"`
	LogLevel     string        `mapstructure:"log_level"`
}

// BackendConfig specifies a target Triton server instance in a GPU pool.
type BackendConfig struct {
	Address       string `mapstructure:"address"`
	NodePool      string `mapstructure:"node_pool"`
	GPUType       string `mapstructure:"gpu_type"`
	MaxConcurrent int    `mapstructure:"max_concurrent"`
	Weight        int    `mapstructure:"weight"`
}

// LoadBalanceConfig tunes the latency-aware balancing algorithm.
type LoadBalanceConfig struct {
	Algorithm string  `mapstructure:"algorithm"` // latency_aware, round_robin, weighted_random
	Alpha     float64 `mapstructure:"alpha"`     // weight for active requests
	Beta      float64 `mapstructure:"beta"`      // weight for moving average latency
	Gamma     float64 `mapstructure:"gamma"`     // weight for queue depth
}

// QueueConfig manages priority queue limits and eviction timeouts.
type QueueConfig struct {
	MaxDepth       int           `mapstructure:"max_depth"`
	DefaultTimeout time.Duration `mapstructure:"default_timeout"`
}

// RedisConfig options for caching routing decisions and model availability.
type RedisConfig struct {
	Enabled  bool          `mapstructure:"enabled"`
	Address  string        `mapstructure:"address"`
	Password string        `mapstructure:"password"`
	DB       int           `mapstructure:"db"`
	TTL      time.Duration `mapstructure:"ttl"`
}

// HealthCheckConfig configures proactive backend liveness and readiness polling.
type HealthCheckConfig struct {
	Interval           time.Duration `mapstructure:"interval"`
	Timeout            time.Duration `mapstructure:"timeout"`
	UnhealthyThreshold int           `mapstructure:"unhealthy_threshold"`
	HealthyThreshold   int           `mapstructure:"healthy_threshold"`
}

// Load reads configuration from file with environment variable overrides.
func Load(configPath string) (*Config, error) {
	v := viper.New()

	if configPath != "" {
		v.SetConfigFile(configPath)
	} else {
		v.SetConfigName("config")
		v.SetConfigType("yaml")
		v.AddConfigPath(".")
		v.AddConfigPath("/etc/inference-router")
	}

	v.SetEnvPrefix("ROUTER")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	setDefaults(v)

	if err := v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config structure: %w", err)
	}

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return &cfg, nil
}

func setDefaults(v *viper.Viper) {
	v.SetDefault("server.grpc_port", 50051)
	v.SetDefault("server.http_port", 8080)
	v.SetDefault("server.metrics_port", 9090)
	v.SetDefault("server.read_timeout", 10*time.Second)
	v.SetDefault("server.write_timeout", 10*time.Second)
	v.SetDefault("server.log_level", "info")

	v.SetDefault("load_balancing.algorithm", "latency_aware")
	v.SetDefault("load_balancing.alpha", 0.4)
	v.SetDefault("load_balancing.beta", 0.4)
	v.SetDefault("load_balancing.gamma", 0.2)

	v.SetDefault("queue.max_depth", 5000)
	v.SetDefault("queue.default_timeout", 5*time.Second)

	v.SetDefault("redis.enabled", false)
	v.SetDefault("redis.address", "localhost:6379")
	v.SetDefault("redis.db", 0)
	v.SetDefault("redis.ttl", 60*time.Second)

	v.SetDefault("health_check.interval", 5*time.Second)
	v.SetDefault("health_check.timeout", 2*time.Second)
	v.SetDefault("health_check.unhealthy_threshold", 3)
	v.SetDefault("health_check.healthy_threshold", 2)
}

// Validate checks sanity of loaded settings.
func (c *Config) Validate() error {
	if c.Server.GRPCPort <= 0 || c.Server.GRPCPort > 65535 {
		return fmt.Errorf("invalid grpc_port: %d", c.Server.GRPCPort)
	}
	if c.Server.HTTPPort <= 0 || c.Server.HTTPPort > 65535 {
		return fmt.Errorf("invalid http_port: %d", c.Server.HTTPPort)
	}
	if len(c.Backends) == 0 {
		return fmt.Errorf("at least one backend must be configured")
	}
	for i, b := range c.Backends {
		if b.Address == "" {
			return fmt.Errorf("backend[%d] missing address", i)
		}
	}
	return nil
}

