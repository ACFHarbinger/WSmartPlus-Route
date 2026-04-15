# {py:mod}`src.configs.policies.lb`

```{py:module} src.configs.policies.lb
```

```{autodoc2-docstring} src.configs.policies.lb
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LocalBranchingConfig <src.configs.policies.lb.LocalBranchingConfig>`
  - ```{autodoc2-docstring} src.configs.policies.lb.LocalBranchingConfig
    :summary:
    ```
````

### API

`````{py:class} LocalBranchingConfig
:canonical: src.configs.policies.lb.LocalBranchingConfig

```{autodoc2-docstring} src.configs.policies.lb.LocalBranchingConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.lb.LocalBranchingConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.lb.LocalBranchingConfig.time_limit
```

````

````{py:attribute} time_limit_per_iteration
:canonical: src.configs.policies.lb.LocalBranchingConfig.time_limit_per_iteration
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.policies.lb.LocalBranchingConfig.time_limit_per_iteration
```

````

````{py:attribute} k
:canonical: src.configs.policies.lb.LocalBranchingConfig.k
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.lb.LocalBranchingConfig.k
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.lb.LocalBranchingConfig.max_iterations
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.lb.LocalBranchingConfig.max_iterations
```

````

````{py:attribute} node_limit_per_iteration
:canonical: src.configs.policies.lb.LocalBranchingConfig.node_limit_per_iteration
:type: int
:value: >
   5000

```{autodoc2-docstring} src.configs.policies.lb.LocalBranchingConfig.node_limit_per_iteration
```

````

````{py:attribute} mip_gap
:canonical: src.configs.policies.lb.LocalBranchingConfig.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.lb.LocalBranchingConfig.mip_gap
```

````

````{py:attribute} seed
:canonical: src.configs.policies.lb.LocalBranchingConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.lb.LocalBranchingConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.lb.LocalBranchingConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.lb.LocalBranchingConfig.vrpp
```

````

````{py:attribute} engine
:canonical: src.configs.policies.lb.LocalBranchingConfig.engine
:type: str
:value: >
   'gurobi'

```{autodoc2-docstring} src.configs.policies.lb.LocalBranchingConfig.engine
```

````

````{py:attribute} framework
:canonical: src.configs.policies.lb.LocalBranchingConfig.framework
:type: str
:value: >
   'ortools'

```{autodoc2-docstring} src.configs.policies.lb.LocalBranchingConfig.framework
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.lb.LocalBranchingConfig.must_go
:type: typing.Optional[src.configs.policies.other.must_go.MustGoConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.lb.LocalBranchingConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.lb.LocalBranchingConfig.post_processing
:type: typing.Optional[src.configs.policies.other.post_processing.PostProcessingConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.lb.LocalBranchingConfig.post_processing
```

````

`````
