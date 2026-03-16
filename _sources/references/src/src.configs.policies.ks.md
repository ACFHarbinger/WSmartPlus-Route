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

````{py:attribute} engine
:canonical: src.configs.policies.ks.KernelSearchConfig.engine
:type: str
:value: >
   'custom'

```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig.engine
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.ks.KernelSearchConfig.must_go
:type: typing.Optional[src.configs.policies.other.must_go.MustGoConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.ks.KernelSearchConfig.post_processing
:type: typing.Optional[src.configs.policies.other.post_processing.PostProcessingConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ks.KernelSearchConfig.post_processing
```

````

`````
