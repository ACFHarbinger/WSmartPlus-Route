# {py:mod}`src.policies.route_construction.matheuristics.local_branching.params`

```{py:module} src.policies.route_construction.matheuristics.local_branching.params
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LBParams <src.policies.route_construction.matheuristics.local_branching.params.LBParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching.params.LBParams
    :summary:
    ```
````

### API

`````{py:class} LBParams
:canonical: src.policies.route_construction.matheuristics.local_branching.params.LBParams

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching.params.LBParams
```

````{py:attribute} k
:canonical: src.policies.route_construction.matheuristics.local_branching.params.LBParams.k
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching.params.LBParams.k
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.matheuristics.local_branching.params.LBParams.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching.params.LBParams.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.matheuristics.local_branching.params.LBParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching.params.LBParams.time_limit
```

````

````{py:attribute} time_limit_per_iteration
:canonical: src.policies.route_construction.matheuristics.local_branching.params.LBParams.time_limit_per_iteration
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching.params.LBParams.time_limit_per_iteration
```

````

````{py:attribute} node_limit_per_iteration
:canonical: src.policies.route_construction.matheuristics.local_branching.params.LBParams.node_limit_per_iteration
:type: int
:value: >
   1000

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching.params.LBParams.node_limit_per_iteration
```

````

````{py:attribute} mip_gap
:canonical: src.policies.route_construction.matheuristics.local_branching.params.LBParams.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching.params.LBParams.mip_gap
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.matheuristics.local_branching.params.LBParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching.params.LBParams.seed
```

````

````{py:attribute} engine
:canonical: src.policies.route_construction.matheuristics.local_branching.params.LBParams.engine
:type: str
:value: >
   'gurobi'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching.params.LBParams.engine
```

````

````{py:attribute} framework
:canonical: src.policies.route_construction.matheuristics.local_branching.params.LBParams.framework
:type: str
:value: >
   'ortools'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching.params.LBParams.framework
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.matheuristics.local_branching.params.LBParams
:canonical: src.policies.route_construction.matheuristics.local_branching.params.LBParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching.params.LBParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.matheuristics.local_branching.params.LBParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching.params.LBParams.to_dict
```

````

`````
