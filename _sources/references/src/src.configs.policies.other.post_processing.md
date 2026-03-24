# {py:mod}`src.configs.policies.other.post_processing`

```{py:module} src.configs.policies.other.post_processing
```

```{autodoc2-docstring} src.configs.policies.other.post_processing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FastTSPPostConfig <src.configs.policies.other.post_processing.FastTSPPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.FastTSPPostConfig
    :summary:
    ```
* - {py:obj}`LKHPostConfig <src.configs.policies.other.post_processing.LKHPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.LKHPostConfig
    :summary:
    ```
* - {py:obj}`LocalSearchPostConfig <src.configs.policies.other.post_processing.LocalSearchPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.LocalSearchPostConfig
    :summary:
    ```
* - {py:obj}`PathPostConfig <src.configs.policies.other.post_processing.PathPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.PathPostConfig
    :summary:
    ```
* - {py:obj}`RandomLocalSearchPostConfig <src.configs.policies.other.post_processing.RandomLocalSearchPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.RandomLocalSearchPostConfig
    :summary:
    ```
* - {py:obj}`PostProcessingConfig <src.configs.policies.other.post_processing.PostProcessingConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig
    :summary:
    ```
````

### API

`````{py:class} FastTSPPostConfig
:canonical: src.configs.policies.other.post_processing.FastTSPPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.FastTSPPostConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.other.post_processing.FastTSPPostConfig.time_limit
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.FastTSPPostConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.FastTSPPostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.FastTSPPostConfig.seed
```

````

`````

`````{py:class} LKHPostConfig
:canonical: src.configs.policies.other.post_processing.LKHPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.LKHPostConfig
```

````{py:attribute} max_iterations
:canonical: src.configs.policies.other.post_processing.LKHPostConfig.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.other.post_processing.LKHPostConfig.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.other.post_processing.LKHPostConfig.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.LKHPostConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.LKHPostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.LKHPostConfig.seed
```

````

`````

`````{py:class} LocalSearchPostConfig
:canonical: src.configs.policies.other.post_processing.LocalSearchPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.LocalSearchPostConfig
```

````{py:attribute} ls_operator
:canonical: src.configs.policies.other.post_processing.LocalSearchPostConfig.ls_operator
:type: str
:value: >
   '2opt'

```{autodoc2-docstring} src.configs.policies.other.post_processing.LocalSearchPostConfig.ls_operator
```

````

````{py:attribute} iterations
:canonical: src.configs.policies.other.post_processing.LocalSearchPostConfig.iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.other.post_processing.LocalSearchPostConfig.iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.other.post_processing.LocalSearchPostConfig.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.LocalSearchPostConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.LocalSearchPostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.LocalSearchPostConfig.seed
```

````

`````

`````{py:class} PathPostConfig
:canonical: src.configs.policies.other.post_processing.PathPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.PathPostConfig
```

````{py:attribute} vehicle_capacity
:canonical: src.configs.policies.other.post_processing.PathPostConfig.vehicle_capacity
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.PathPostConfig.vehicle_capacity
```

````

`````

`````{py:class} RandomLocalSearchPostConfig
:canonical: src.configs.policies.other.post_processing.RandomLocalSearchPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.RandomLocalSearchPostConfig
```

````{py:attribute} iterations
:canonical: src.configs.policies.other.post_processing.RandomLocalSearchPostConfig.iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.other.post_processing.RandomLocalSearchPostConfig.iterations
```

````

````{py:attribute} params
:canonical: src.configs.policies.other.post_processing.RandomLocalSearchPostConfig.params
:type: typing.Dict[str, float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.RandomLocalSearchPostConfig.params
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.other.post_processing.RandomLocalSearchPostConfig.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.RandomLocalSearchPostConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.RandomLocalSearchPostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.RandomLocalSearchPostConfig.seed
```

````

`````

`````{py:class} PostProcessingConfig
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig
```

````{py:attribute} methods
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.methods
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.methods
```

````

````{py:attribute} fast_tsp
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.fast_tsp
:type: src.configs.policies.other.post_processing.FastTSPPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.fast_tsp
```

````

````{py:attribute} lkh
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.lkh
:type: src.configs.policies.other.post_processing.LKHPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.lkh
```

````

````{py:attribute} local_search
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.local_search
:type: src.configs.policies.other.post_processing.LocalSearchPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.local_search
```

````

````{py:attribute} random_local_search
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.random_local_search
:type: src.configs.policies.other.post_processing.RandomLocalSearchPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.random_local_search
```

````

````{py:attribute} path
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.path
:type: src.configs.policies.other.post_processing.PathPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.path
```

````

````{py:attribute} params
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.params
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.params
```

````

`````
