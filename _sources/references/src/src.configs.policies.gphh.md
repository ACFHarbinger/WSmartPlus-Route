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

````{py:attribute} engine
:canonical: src.configs.policies.gphh.GPHHConfig.engine
:type: str
:value: >
   'gphh'

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.engine
```

````

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

````{py:attribute} eval_steps
:canonical: src.configs.policies.gphh.GPHHConfig.eval_steps
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.eval_steps
```

````

````{py:attribute} apply_steps
:canonical: src.configs.policies.gphh.GPHHConfig.apply_steps
:type: int
:value: >
   200

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.apply_steps
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

````{py:attribute} n_llh
:canonical: src.configs.policies.gphh.GPHHConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.n_llh
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.gphh.GPHHConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.n_removal
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

````{py:attribute} vrpp
:canonical: src.configs.policies.gphh.GPHHConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.gphh.GPHHConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.gphh.GPHHConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.gphh.GPHHConfig.post_processing
```

````

`````
