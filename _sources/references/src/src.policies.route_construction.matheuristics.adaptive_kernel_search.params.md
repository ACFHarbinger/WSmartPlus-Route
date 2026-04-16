# {py:mod}`src.policies.route_construction.matheuristics.adaptive_kernel_search.params`

```{py:module} src.policies.route_construction.matheuristics.adaptive_kernel_search.params
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.adaptive_kernel_search.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AKSParams <src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams
    :summary:
    ```
````

### API

`````{py:class} AKSParams
:canonical: src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams

```{autodoc2-docstring} src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams
```

````{py:attribute} initial_kernel_size
:canonical: src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.initial_kernel_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.initial_kernel_size
```

````

````{py:attribute} bucket_size
:canonical: src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.bucket_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.bucket_size
```

````

````{py:attribute} max_buckets
:canonical: src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.max_buckets
:type: int
:value: >
   15

```{autodoc2-docstring} src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.max_buckets
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.time_limit
```

````

````{py:attribute} mip_limit_nodes
:canonical: src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.mip_limit_nodes
:type: int
:value: >
   10000

```{autodoc2-docstring} src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.mip_limit_nodes
```

````

````{py:attribute} mip_gap
:canonical: src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.mip_gap
```

````

````{py:attribute} t_easy
:canonical: src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.t_easy
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.t_easy
```

````

````{py:attribute} epsilon
:canonical: src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.epsilon
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.epsilon
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.seed
```

````

````{py:attribute} engine
:canonical: src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.engine
:type: str
:value: >
   'gurobi'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.engine
```

````

````{py:attribute} framework
:canonical: src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.framework
:type: str
:value: >
   'ortools'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.framework
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams
:canonical: src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.matheuristics.adaptive_kernel_search.params.AKSParams.to_dict
```

````

`````
