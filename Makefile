# =============================================================================
# Makefile - Distributed GPU Inference Platform
# =============================================================================

.DEFAULT_GOAL := help

REGISTRY ?= distributed-gpu-inference
TAG ?= latest
GO_FILES := $(shell find . -name '*.go' -not -path './vendor/*')

.PHONY: help
help: ## Show this help message
	@echo "Distributed GPU Inference Platform Build System"
	@echo "================================================"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: build
build: ## Compile Go inference router and CLI binaries
	@echo ">>> Building Go router and CLI binaries..."
	CGO_ENABLED=0 go build -ldflags="-w -s" -o bin/inference-router cmd/router/main.go
	CGO_ENABLED=0 go build -ldflags="-w -s" -o bin/dist-inf cmd/cli/main.go

.PHONY: demo
demo: ## Run automated end-to-end platform demo (cluster simulation + chaos)
	@./scripts/demo.sh

.PHONY: run
run: ## Run Go router locally with default config
	@echo ">>> Running Go router locally..."
	go run cmd/router/main.go --config config.yaml

.PHONY: proto
proto: ## Generate Go protobuf and gRPC bindings from proto definitions
	@echo ">>> Compiling Protocol Buffers..."
	protoc --proto_path=. \
		--go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		pkg/proto/inference.proto pkg/proto/health.proto

.PHONY: test-go
test-go: ## Run Go unit and benchmark tests
	@echo ">>> Running Go tests..."
	go test -v -race ./...

.PHONY: test-python
test-python: ## Run Python pipeline unit and integration tests
	@echo ">>> Running Python tests..."
	pytest -v python/tests/

.PHONY: test
test: test-go test-python ## Run all tests

.PHONY: docker-build
docker-build: ## Build all Docker container images
	@echo ">>> Building Docker images..."
	docker build -t $(REGISTRY)/router:$(TAG) -f docker/Dockerfile.router .
	docker build -t $(REGISTRY)/pipeline:$(TAG) -f docker/Dockerfile.pipeline .
	docker build -t $(REGISTRY)/triton-custom:$(TAG) -f docker/Dockerfile.triton .

.PHONY: setup-cluster
setup-cluster: ## Bootstrap multi-node Kubernetes cluster (Kind)
	@./scripts/setup-cluster.sh kind

.PHONY: install-gpu-operator
install-gpu-operator: ## Deploy NVIDIA GPU Operator via Helm
	@./scripts/install-gpu-operator.sh

.PHONY: deploy-dev
deploy-dev: ## Deploy to local/dev Kubernetes cluster
	@./scripts/deploy.sh dev

.PHONY: deploy-prod
deploy-prod: ## Deploy to production Kubernetes cluster
	@./scripts/deploy.sh prod

.PHONY: mock-cluster
mock-cluster: ## Run simulated 4-pool Triton cluster locally
	@python3 scripts/mock_triton_cluster.py --base-port 8100

.PHONY: simulate
simulate: ## Run peak inference simulation against mock cluster or router
	@python3 scripts/simulate_inference.py --concurrency 20 --requests 300

.PHONY: load-test
load-test: ## Run gRPC load testing benchmark against router
	@./scripts/load-test.sh localhost:50051 50 5000

.PHONY: chaos-test
chaos-test: ## Run automated chaos engineering test suite
	@python3 scripts/chaos_test.py --router-url http://localhost:8080

.PHONY: vllm-serve
vllm-serve: ## Launch local vLLM continuous batching LLM server
	@python3 python/model_pipeline/serve_vllm.py

.PHONY: dashboard
dashboard: ## Open the interactive cluster visualizer console in browser
	@open http://localhost:8080/ || echo "Open http://localhost:8080/ in your browser"

.PHONY: shard-data
shard-data: ## Execute sample dataset sharding pipeline
	@python3 -m python.data_pipeline.sharding \
		--input test_data.parquet \
		--output /tmp/shards \
		--num-shards 8

.PHONY: clean
clean: ## Clean build binaries and temp files
	@echo ">>> Cleaning build artifacts..."
	rm -rf bin/ /tmp/inference_cache /tmp/shards
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

