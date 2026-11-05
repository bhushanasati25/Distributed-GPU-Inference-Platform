# Contributing to Distributed GPU Inference Platform

We welcome contributions from the community to improve the platform's routing algorithms, model optimization pipelines, and Kubernetes infrastructure.

---

## Development Workflow

1. **Fork & Clone**:
   ```bash
   git clone https://github.com/bhushanasati25/Distributed-GPU-Inference-Platform.git
   cd Distributed-GPU-Inference-Platform
   ```

2. **Run Local Verification**:
   ```bash
   # Run all Go and Python tests
   make test

   # Run automated platform demo
   make demo
   ```

3. **Branching Strategy**:
   - `feat/your-feature-name` for new features.
   - `fix/issue-description` for bug fixes.
   - `docs/update-description` for documentation improvements.

4. **Coding Standards**:
   - Go: Format with `gofmt -s -w .` and check with `golangci-lint run`.
   - Python: Format with `black` and follow PEP 8 with Google-style docstrings.
   - Kubernetes: Validate manifests with `kubectl kustomize`.

5. **Submitting a Pull Request**:
   - Ensure all CI tests pass.
   - Link any relevant issues.
   - Add appropriate unit/integration tests for your changes.
