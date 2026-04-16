# {py:mod}`src.configs.policies.gphh`

```{py:module} src.configs.policies.gphh
```

```{autodoc2-docstring} src.configs.policies.gphh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GPHHConfig <src.configs.policies.gphh.GPHHConfig>`
  - ```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig
    :summary:
    ```
````

### API

`````{py:class} GPHHConfig
:canonical: src.configs.policies.gphh.GPHHConfig

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig
```

````{py:attribute} gp_pop_size
:canonical: src.configs.policies.gphh.GPHHConfig.gp_pop_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.gp_pop_size
```

````

````{py:attribute} max_gp_generations
:canonical: src.configs.policies.gphh.GPHHConfig.max_gp_generations
:type: int
:value: >
   30

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.max_gp_generations
```

````

````{py:attribute} tree_depth
:canonical: src.configs.policies.gphh.GPHHConfig.tree_depth
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.tree_depth
```

````

````{py:attribute} tournament_size
:canonical: src.configs.policies.gphh.GPHHConfig.tournament_size
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.tournament_size
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.gphh.GPHHConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.time_limit
```

````

````{py:attribute} parsimony_coefficient
:canonical: src.configs.policies.gphh.GPHHConfig.parsimony_coefficient
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.parsimony_coefficient
```

````

````{py:attribute} n_training_instances
:canonical: src.configs.policies.gphh.GPHHConfig.n_training_instances
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.n_training_instances
```

````

````{py:attribute} training_sample_ratio
:canonical: src.configs.policies.gphh.GPHHConfig.training_sample_ratio
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.training_sample_ratio
```

````

````{py:attribute} seed
:canonical: src.configs.policies.gphh.GPHHConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.gphh.GPHHConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.vrpp
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.gphh.GPHHConfig.mandatory_selection
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.gphh.GPHHConfig.route_improvement
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.route_improvement
```

````

`````
