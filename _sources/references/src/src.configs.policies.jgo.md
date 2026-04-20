# {py:mod}`src.configs.policies.jgo`

```{py:module} src.configs.policies.jgo
```

```{autodoc2-docstring} src.configs.policies.jgo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointGreedyConfig <src.configs.policies.jgo.JointGreedyConfig>`
  - ```{autodoc2-docstring} src.configs.policies.jgo.JointGreedyConfig
    :summary:
    ```
````

### API

`````{py:class} JointGreedyConfig
:canonical: src.configs.policies.jgo.JointGreedyConfig

```{autodoc2-docstring} src.configs.policies.jgo.JointGreedyConfig
```

````{py:attribute} k_best
:canonical: src.configs.policies.jgo.JointGreedyConfig.k_best
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.jgo.JointGreedyConfig.k_best
```

````

````{py:attribute} n_starts
:canonical: src.configs.policies.jgo.JointGreedyConfig.n_starts
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.jgo.JointGreedyConfig.n_starts
```

````

````{py:attribute} distance_weight
:canonical: src.configs.policies.jgo.JointGreedyConfig.distance_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.jgo.JointGreedyConfig.distance_weight
```

````

````{py:attribute} seed
:canonical: src.configs.policies.jgo.JointGreedyConfig.seed
:type: typing.Optional[int]
:value: >
   42

```{autodoc2-docstring} src.configs.policies.jgo.JointGreedyConfig.seed
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.jgo.JointGreedyConfig.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.policies.jgo.JointGreedyConfig.time_limit
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.jgo.JointGreedyConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.jgo.JointGreedyConfig.vrpp
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.jgo.JointGreedyConfig.mandatory_selection
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.jgo.JointGreedyConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.jgo.JointGreedyConfig.route_improvement
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.jgo.JointGreedyConfig.route_improvement
```

````

`````
