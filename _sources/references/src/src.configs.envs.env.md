# {py:mod}`src.configs.envs.env`

```{py:module} src.configs.envs.env
```

```{autodoc2-docstring} src.configs.envs.env
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EnvConfig <src.configs.envs.env.EnvConfig>`
  - ```{autodoc2-docstring} src.configs.envs.env.EnvConfig
    :summary:
    ```
````

### API

`````{py:class} EnvConfig
:canonical: src.configs.envs.env.EnvConfig

```{autodoc2-docstring} src.configs.envs.env.EnvConfig
```

````{py:attribute} name
:canonical: src.configs.envs.env.EnvConfig.name
:type: str
:value: >
   'vrpp'

```{autodoc2-docstring} src.configs.envs.env.EnvConfig.name
```

````

````{py:attribute} min_loc
:canonical: src.configs.envs.env.EnvConfig.min_loc
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.envs.env.EnvConfig.min_loc
```

````

````{py:attribute} max_loc
:canonical: src.configs.envs.env.EnvConfig.max_loc
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.envs.env.EnvConfig.max_loc
```

````

````{py:attribute} graph
:canonical: src.configs.envs.env.EnvConfig.graph
:type: src.configs.envs.graph.GraphConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.envs.env.EnvConfig.graph
```

````

````{py:attribute} reward
:canonical: src.configs.envs.env.EnvConfig.reward
:type: src.configs.envs.objective.ObjectiveConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.envs.env.EnvConfig.reward
```

````

````{py:attribute} data_distribution
:canonical: src.configs.envs.env.EnvConfig.data_distribution
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.envs.env.EnvConfig.data_distribution
```

````

````{py:attribute} min_fill
:canonical: src.configs.envs.env.EnvConfig.min_fill
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.envs.env.EnvConfig.min_fill
```

````

````{py:attribute} max_fill
:canonical: src.configs.envs.env.EnvConfig.max_fill
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.envs.env.EnvConfig.max_fill
```

````

````{py:attribute} fill_distribution
:canonical: src.configs.envs.env.EnvConfig.fill_distribution
:type: str
:value: >
   'uniform'

```{autodoc2-docstring} src.configs.envs.env.EnvConfig.fill_distribution
```

````

`````
