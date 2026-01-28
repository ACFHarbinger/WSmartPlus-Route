# Advanced Testing, Benchmarking, and Logging Framework

This document describes the advanced testing suite and logging infrastructure implemented for WSmart-Route.

## 1. Mutation Testing

Mutation testing identifies gaps in the test suite by injecting small bugs (mutations) into the source code and checking if existing tests fail.

- **Tool**: `mutmut`
- **Configuration**: `mutmut_config.py` (filters core logic in `logic/src/`)
- **Commands**:
  - `just mutation-test`: Run mutation tests.
  - `just mutation-report`: View detailed results.

## 2. Performance Benchmarking

A formalized suite to track solver latency, throughput, and solution quality.

- **Scripts**: `logic/benchmark/run_all.py` (centralized entry point)
  - Individual benchmarks: `baseline_benchmarks.py`, `neural_benchmarks.py`, `benchmark_policies.py`, etc.
- **Command**: `just benchmark`

## 3. Solver Contract Tests

Ensures parity and robustness across different optimization engines.

- **Location**: `logic/test/integration/test_solver_contracts.py`
- **Features**:
  - Parity checks: Ensures Gurobi and Hexaly produce similar results formatted identically.
  - Interface validation: Validates input/output schemas of solver functions.
  - Edge cases: Tests stability with empty instances and invalid IDs.

## 4. ELK Stack Logging

Structured logging infrastructure for visualizing test metrics and benchmarking results.

- **Infrastructure**: Docker Compose in `docker/elk/`.
- **Structured Logs**: `logic/src/utils/structured_logging.py` provides JSON formatting for Logstash.
- **Usage**:
  ```python
  from logic.src.utils.structured_logging import log_test_metric
  log_test_metric("inference_latency", 12.5)
  ```
- **Kibana**: Dashboard template available at `docker/elk/kibana_dashboard.json`.

---

To start the ELK stack:
```bash
cd docker/elk
docker-compose up -d
```
The dashboard will be available at `http://localhost:5601`.
