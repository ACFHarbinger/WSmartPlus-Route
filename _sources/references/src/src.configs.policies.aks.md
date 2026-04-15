# {py:mod}`src.configs.policies.aks`

```{py:module} src.configs.policies.aks
```

```{autodoc2-docstring} src.configs.policies.aks
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AdaptiveKernelSearchConfig <src.configs.policies.aks.AdaptiveKernelSearchConfig>`
  - ```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig
    :summary:
    ```
````

### API

`````{py:class} AdaptiveKernelSearchConfig
:canonical: src.configs.policies.aks.AdaptiveKernelSearchConfig

```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.aks.AdaptiveKernelSearchConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig.time_limit
```

````

````{py:attribute} initial_kernel_size
:canonical: src.configs.policies.aks.AdaptiveKernelSearchConfig.initial_kernel_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig.initial_kernel_size
```

````

````{py:attribute} bucket_size
:canonical: src.configs.policies.aks.AdaptiveKernelSearchConfig.bucket_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig.bucket_size
```

````

````{py:attribute} max_buckets
:canonical: src.configs.policies.aks.AdaptiveKernelSearchConfig.max_buckets
:type: int
:value: >
   15

```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig.max_buckets
```

````

````{py:attribute} mip_limit_nodes
:canonical: src.configs.policies.aks.AdaptiveKernelSearchConfig.mip_limit_nodes
:type: int
:value: >
   10000

```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig.mip_limit_nodes
```

````

````{py:attribute} mip_gap
:canonical: src.configs.policies.aks.AdaptiveKernelSearchConfig.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig.mip_gap
```

````

````{py:attribute} seed
:canonical: src.configs.policies.aks.AdaptiveKernelSearchConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.aks.AdaptiveKernelSearchConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig.vrpp
```

````

````{py:attribute} t_easy
:canonical: src.configs.policies.aks.AdaptiveKernelSearchConfig.t_easy
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig.t_easy
```

````

````{py:attribute} epsilon
:canonical: src.configs.policies.aks.AdaptiveKernelSearchConfig.epsilon
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig.epsilon
```

````

````{py:attribute} time_limit_stage_1
:canonical: src.configs.policies.aks.AdaptiveKernelSearchConfig.time_limit_stage_1
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig.time_limit_stage_1
```

````

````{py:attribute} engine
:canonical: src.configs.policies.aks.AdaptiveKernelSearchConfig.engine
:type: str
:value: >
   'gurobi'

```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig.engine
```

````

````{py:attribute} framework
:canonical: src.configs.policies.aks.AdaptiveKernelSearchConfig.framework
:type: str
:value: >
   'ortools'

```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig.framework
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.aks.AdaptiveKernelSearchConfig.must_go
:type: typing.Optional[src.configs.policies.other.must_go.MustGoConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.aks.AdaptiveKernelSearchConfig.post_processing
:type: typing.Optional[src.configs.policies.other.post_processing.PostProcessingConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.aks.AdaptiveKernelSearchConfig.post_processing
```

````

`````
