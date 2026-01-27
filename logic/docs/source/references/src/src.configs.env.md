# {py:mod}`src.configs.env`

```{py:module} src.configs.env
```

```{autodoc2-docstring} src.configs.env
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EnvConfig <src.configs.env.EnvConfig>`
  - ```{autodoc2-docstring} src.configs.env.EnvConfig
    :summary:
    ```
````

### API

`````{py:class} EnvConfig
:canonical: src.configs.env.EnvConfig

```{autodoc2-docstring} src.configs.env.EnvConfig
```

````{py:attribute} name
:canonical: src.configs.env.EnvConfig.name
:type: str
:value: >
   'vrpp'

```{autodoc2-docstring} src.configs.env.EnvConfig.name
```

````

````{py:attribute} num_loc
:canonical: src.configs.env.EnvConfig.num_loc
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.env.EnvConfig.num_loc
```

````

````{py:attribute} min_loc
:canonical: src.configs.env.EnvConfig.min_loc
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.env.EnvConfig.min_loc
```

````

````{py:attribute} max_loc
:canonical: src.configs.env.EnvConfig.max_loc
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.env.EnvConfig.max_loc
```

````

````{py:attribute} capacity
:canonical: src.configs.env.EnvConfig.capacity
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.configs.env.EnvConfig.capacity
```

````

````{py:attribute} overflow_penalty
:canonical: src.configs.env.EnvConfig.overflow_penalty
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.env.EnvConfig.overflow_penalty
```

````

````{py:attribute} collection_reward
:canonical: src.configs.env.EnvConfig.collection_reward
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.env.EnvConfig.collection_reward
```

````

````{py:attribute} cost_weight
:canonical: src.configs.env.EnvConfig.cost_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.env.EnvConfig.cost_weight
```

````

````{py:attribute} prize_weight
:canonical: src.configs.env.EnvConfig.prize_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.env.EnvConfig.prize_weight
```

````

````{py:attribute} area
:canonical: src.configs.env.EnvConfig.area
:type: str
:value: >
   'riomaior'

```{autodoc2-docstring} src.configs.env.EnvConfig.area
```

````

````{py:attribute} waste_type
:canonical: src.configs.env.EnvConfig.waste_type
:type: str
:value: >
   'plastic'

```{autodoc2-docstring} src.configs.env.EnvConfig.waste_type
```

````

````{py:attribute} focus_graph
:canonical: src.configs.env.EnvConfig.focus_graph
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.env.EnvConfig.focus_graph
```

````

````{py:attribute} focus_size
:canonical: src.configs.env.EnvConfig.focus_size
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.env.EnvConfig.focus_size
```

````

````{py:attribute} eval_focus_size
:canonical: src.configs.env.EnvConfig.eval_focus_size
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.env.EnvConfig.eval_focus_size
```

````

````{py:attribute} distance_method
:canonical: src.configs.env.EnvConfig.distance_method
:type: str
:value: >
   'ogd'

```{autodoc2-docstring} src.configs.env.EnvConfig.distance_method
```

````

````{py:attribute} dm_filepath
:canonical: src.configs.env.EnvConfig.dm_filepath
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.env.EnvConfig.dm_filepath
```

````

````{py:attribute} waste_filepath
:canonical: src.configs.env.EnvConfig.waste_filepath
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.env.EnvConfig.waste_filepath
```

````

````{py:attribute} vertex_method
:canonical: src.configs.env.EnvConfig.vertex_method
:type: str
:value: >
   'mmn'

```{autodoc2-docstring} src.configs.env.EnvConfig.vertex_method
```

````

````{py:attribute} edge_threshold
:canonical: src.configs.env.EnvConfig.edge_threshold
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.env.EnvConfig.edge_threshold
```

````

````{py:attribute} edge_method
:canonical: src.configs.env.EnvConfig.edge_method
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.env.EnvConfig.edge_method
```

````

````{py:attribute} data_distribution
:canonical: src.configs.env.EnvConfig.data_distribution
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.env.EnvConfig.data_distribution
```

````

````{py:attribute} min_fill
:canonical: src.configs.env.EnvConfig.min_fill
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.env.EnvConfig.min_fill
```

````

````{py:attribute} max_fill
:canonical: src.configs.env.EnvConfig.max_fill
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.env.EnvConfig.max_fill
```

````

````{py:attribute} fill_distribution
:canonical: src.configs.env.EnvConfig.fill_distribution
:type: str
:value: >
   'uniform'

```{autodoc2-docstring} src.configs.env.EnvConfig.fill_distribution
```

````

`````
