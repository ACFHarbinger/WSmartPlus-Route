# {py:mod}`src.policies.route_construction.matheuristics.two_phase_kernel_search.params`

```{py:module} src.policies.route_construction.matheuristics.two_phase_kernel_search.params
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TPKSParams <src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams
    :summary:
    ```
````

### API

`````{py:class} TPKSParams
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams
```

````{py:attribute} phase1_kernel_size
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.phase1_kernel_size
:type: int
:value: >
   30

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.phase1_kernel_size
```

````

````{py:attribute} phase1_bucket_size
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.phase1_bucket_size
:type: int
:value: >
   15

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.phase1_bucket_size
```

````

````{py:attribute} phase1_time_fraction
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.phase1_time_fraction
:type: float
:value: >
   0.35

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.phase1_time_fraction
```

````

````{py:attribute} phase1_mip_node_limit
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.phase1_mip_node_limit
:type: int
:value: >
   2000

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.phase1_mip_node_limit
```

````

````{py:attribute} phase2_bucket_size_easy
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.phase2_bucket_size_easy
:type: int
:value: >
   30

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.phase2_bucket_size_easy
```

````

````{py:attribute} phase2_bucket_size_normal
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.phase2_bucket_size_normal
:type: int
:value: >
   15

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.phase2_bucket_size_normal
```

````

````{py:attribute} phase2_time_limit_per_bucket
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.phase2_time_limit_per_bucket
:type: float
:value: >
   15.0

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.phase2_time_limit_per_bucket
```

````

````{py:attribute} max_buckets
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.max_buckets
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.max_buckets
```

````

````{py:attribute} t_easy
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.t_easy
:type: float
:value: >
   8.0

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.t_easy
```

````

````{py:attribute} epsilon
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.epsilon
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.epsilon
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.time_limit
```

````

````{py:attribute} mip_gap
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.mip_gap
```

````

````{py:attribute} mip_limit_nodes
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.mip_limit_nodes
:type: int
:value: >
   10000

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.mip_limit_nodes
```

````

````{py:attribute} initial_kernel_size
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.initial_kernel_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.initial_kernel_size
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.seed
```

````

````{py:attribute} engine
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.engine
:type: str
:value: >
   'gurobi'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.engine
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.from_config
```

````

````{py:method} to_dict()
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams.to_dict
```

````

`````
