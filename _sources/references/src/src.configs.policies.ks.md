# {py:mod}`src.configs.policies.ks`

```{py:module} src.configs.policies.ks
```

```{autodoc2-docstring} src.configs.policies.ks
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KernelSearchConfig <src.configs.policies.ks.KernelSearchConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig
    :summary:
    ```
````

### API

`````{py:class} KernelSearchConfig
:canonical: src.configs.policies.ks.KernelSearchConfig

```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.ks.KernelSearchConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig.time_limit
```

````

````{py:attribute} initial_kernel_size
:canonical: src.configs.policies.ks.KernelSearchConfig.initial_kernel_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig.initial_kernel_size
```

````

````{py:attribute} bucket_size
:canonical: src.configs.policies.ks.KernelSearchConfig.bucket_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig.bucket_size
```

````

````{py:attribute} max_buckets
:canonical: src.configs.policies.ks.KernelSearchConfig.max_buckets
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig.max_buckets
```

````

````{py:attribute} mip_limit_nodes
:canonical: src.configs.policies.ks.KernelSearchConfig.mip_limit_nodes
:type: int
:value: >
   5000

```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig.mip_limit_nodes
```

````

````{py:attribute} mip_gap
:canonical: src.configs.policies.ks.KernelSearchConfig.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig.mip_gap
```

````

````{py:attribute} seed
:canonical: src.configs.policies.ks.KernelSearchConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.ks.KernelSearchConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig.vrpp
```

````

````{py:attribute} engine
:canonical: src.configs.policies.ks.KernelSearchConfig.engine
:type: str
:value: >
   'gurobi'

```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig.engine
```

````

````{py:attribute} framework
:canonical: src.configs.policies.ks.KernelSearchConfig.framework
:type: str
:value: >
   'ortools'

```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig.framework
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.ks.KernelSearchConfig.mandatory_selection
:type: typing.Optional[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.ks.KernelSearchConfig.route_improvement
:type: typing.Optional[src.configs.policies.other.route_improvement.RouteImprovingConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig.route_improvement
```

````

`````
