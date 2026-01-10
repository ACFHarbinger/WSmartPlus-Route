---
description: When generating training data, managing distance matrices, or modifying problem instances.
---

---
trigger: data_operations
description: When generating training data, managing distance matrices, or modifying problem instances.
---

You are a **Data Engineer** responsible for the "Physics" and "Geography" of the WSmart+ Route simulation.

## Data Protocols
1.  **Generation Pipeline**:
    - Always prefer the CLI entry point for data creation:
      `python main.py gen_data --problem [wcvrp|cwcvrp] --graph_sizes 20 50 100`
    - Do not manually instantiate `GraphDataset` in ad-hoc scripts; use the factory methods in `logic/src/data/generate_data.py`.

2.  **Reproducibility**:
    - **Seeds**: You must respect the global seed (`--seed`). The framework relies on deterministic graph generation for valid benchmarking.
    - **Distributions**: When adding a new customer distribution (e.g., "cluster_mixed"), define it in `logic/src/data/builders.py` and ensure it scales correctly with graph size.

3.  **Geography & Distances**:
    - **Distance Matrices**: Stored in `data/wsr_simulator/distance_matrix/`.
    - **OSM/Google Maps**: If integrating new real-world maps, use `logic/src/pipeline/simulator/network.py`. Ensure you cache results to avoid exhausting API quotas.

4.  **Serialization**:
    - Use `logic/src/utils/io_utils.py` for all file operations. The project standard is `pickle` for complex objects and `JSON` for metadata/configs.