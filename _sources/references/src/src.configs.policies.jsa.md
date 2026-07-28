# {py:mod}`src.configs.policies.jsa`

```{py:module} src.configs.policies.jsa
```

```{autodoc2-docstring} src.configs.policies.jsa
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointSAConfig <src.configs.policies.jsa.JointSAConfig>`
  - ```{autodoc2-docstring} src.configs.policies.jsa.JointSAConfig
    :summary:
    ```
````

### API

`````{py:class} JointSAConfig
:canonical: src.configs.policies.jsa.JointSAConfig

```{autodoc2-docstring} src.configs.policies.jsa.JointSAConfig
```

````{py:attribute} start_temp
:canonical: src.configs.policies.jsa.JointSAConfig.start_temp
:type: float
:value: >
   1000.0

```{autodoc2-docstring} src.configs.policies.jsa.JointSAConfig.start_temp
```

````

````{py:attribute} cooling_rate
:canonical: src.configs.policies.jsa.JointSAConfig.cooling_rate
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.configs.policies.jsa.JointSAConfig.cooling_rate
```

````

````{py:attribute} max_steps
:canonical: src.configs.policies.jsa.JointSAConfig.max_steps
:type: int
:value: >
   2000

```{autodoc2-docstring} src.configs.policies.jsa.JointSAConfig.max_steps
```

````

````{py:attribute} restart_limit
:canonical: src.configs.policies.jsa.JointSAConfig.restart_limit
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.jsa.JointSAConfig.restart_limit
```

````

````{py:attribute} prob_bit_flip
:canonical: src.configs.policies.jsa.JointSAConfig.prob_bit_flip
:type: float
:value: >
   0.7

```{autodoc2-docstring} src.configs.policies.jsa.JointSAConfig.prob_bit_flip
```

````

````{py:attribute} prob_route_swap
:canonical: src.configs.policies.jsa.JointSAConfig.prob_route_swap
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.configs.policies.jsa.JointSAConfig.prob_route_swap
```

````

````{py:attribute} overflow_penalty
:canonical: src.configs.policies.jsa.JointSAConfig.overflow_penalty
:type: float
:value: >
   1000.0

```{autodoc2-docstring} src.configs.policies.jsa.JointSAConfig.overflow_penalty
```

````

````{py:attribute} seed
:canonical: src.configs.policies.jsa.JointSAConfig.seed
:type: typing.Optional[int]
:value: >
   42

```{autodoc2-docstring} src.configs.policies.jsa.JointSAConfig.seed
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.jsa.JointSAConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.jsa.JointSAConfig.time_limit
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.jsa.JointSAConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.jsa.JointSAConfig.vrpp
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.jsa.JointSAConfig.mandatory_selection
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.jsa.JointSAConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.jsa.JointSAConfig.route_improvement
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.jsa.JointSAConfig.route_improvement
```

````

`````
