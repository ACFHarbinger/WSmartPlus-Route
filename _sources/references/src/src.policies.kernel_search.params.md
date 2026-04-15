# {py:mod}`src.policies.kernel_search.params`

```{py:module} src.policies.kernel_search.params
```

```{autodoc2-docstring} src.policies.kernel_search.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KSParams <src.policies.kernel_search.params.KSParams>`
  - ```{autodoc2-docstring} src.policies.kernel_search.params.KSParams
    :summary:
    ```
````

### API

`````{py:class} KSParams
:canonical: src.policies.kernel_search.params.KSParams

```{autodoc2-docstring} src.policies.kernel_search.params.KSParams
```

````{py:attribute} initial_kernel_size
:canonical: src.policies.kernel_search.params.KSParams.initial_kernel_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.kernel_search.params.KSParams.initial_kernel_size
```

````

````{py:attribute} bucket_size
:canonical: src.policies.kernel_search.params.KSParams.bucket_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.kernel_search.params.KSParams.bucket_size
```

````

````{py:attribute} max_buckets
:canonical: src.policies.kernel_search.params.KSParams.max_buckets
:type: int
:value: >
   15

```{autodoc2-docstring} src.policies.kernel_search.params.KSParams.max_buckets
```

````

````{py:attribute} time_limit
:canonical: src.policies.kernel_search.params.KSParams.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.policies.kernel_search.params.KSParams.time_limit
```

````

````{py:attribute} mip_limit_nodes
:canonical: src.policies.kernel_search.params.KSParams.mip_limit_nodes
:type: int
:value: >
   10000

```{autodoc2-docstring} src.policies.kernel_search.params.KSParams.mip_limit_nodes
```

````

````{py:attribute} mip_gap
:canonical: src.policies.kernel_search.params.KSParams.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.kernel_search.params.KSParams.mip_gap
```

````

````{py:attribute} seed
:canonical: src.policies.kernel_search.params.KSParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.kernel_search.params.KSParams.seed
```

````

````{py:attribute} engine
:canonical: src.policies.kernel_search.params.KSParams.engine
:type: str
:value: >
   'gurobi'

```{autodoc2-docstring} src.policies.kernel_search.params.KSParams.engine
```

````

````{py:attribute} framework
:canonical: src.policies.kernel_search.params.KSParams.framework
:type: str
:value: >
   'ortools'

```{autodoc2-docstring} src.policies.kernel_search.params.KSParams.framework
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.kernel_search.params.KSParams
:canonical: src.policies.kernel_search.params.KSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.kernel_search.params.KSParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.kernel_search.params.KSParams.to_dict

```{autodoc2-docstring} src.policies.kernel_search.params.KSParams.to_dict
```

````

`````
