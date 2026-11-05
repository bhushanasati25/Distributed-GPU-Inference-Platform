## Description
Briefly describe the change, motivation, and link related issues.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Performance optimization (routing, batching, caching)
- [ ] Infrastructure / Kubernetes / Helm updates
- [ ] Documentation update

## Verification & Testing
- [ ] `make test-go` passes
- [ ] `make test-python` passes
- [ ] `kubectl kustomize k8s/overlays/prod` validates cleanly
- [ ] Tested with `make demo` or `scripts/simulate_inference.py`

## Checklist
- [ ] Code adheres to Go / Python style standards.
- [ ] Unit tests added for new logic.
- [ ] Documentation updated in `README.md` or `docs/`.

