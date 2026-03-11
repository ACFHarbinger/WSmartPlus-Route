# {py:mod}`src.configs.policies.gihh`

```{py:module} src.configs.policies.gihh
```

```{autodoc2-docstring} src.configs.policies.gihh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GIHHConfig <src.configs.policies.gihh.GIHHConfig>`
  - ```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig
    :summary:
    ```
````

### API

`````{py:class} GIHHConfig
:canonical: src.configs.policies.gihh.GIHHConfig

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.gihh.GIHHConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.gihh.GIHHConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.seed
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.gihh.GIHHConfig.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.max_iterations
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.gihh.GIHHConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.vrpp
```

````

````{py:attribute} move_operators
:canonical: src.configs.policies.gihh.GIHHConfig.move_operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.move_operators
```

````

````{py:attribute} perturbation_operators
:canonical: src.configs.policies.gihh.GIHHConfig.perturbation_operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.perturbation_operators
```

````

````{py:attribute} iri_weight
:canonical: src.configs.policies.gihh.GIHHConfig.iri_weight
:type: float
:value: >
   0.6

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.iri_weight
```

````

````{py:attribute} tbi_weight
:canonical: src.configs.policies.gihh.GIHHConfig.tbi_weight
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.tbi_weight
```

````

````{py:attribute} learning_rate
:canonical: src.configs.policies.gihh.GIHHConfig.learning_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.learning_rate
```

````

````{py:attribute} memory_size
:canonical: src.configs.policies.gihh.GIHHConfig.memory_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.memory_size
```

````

````{py:attribute} epsilon
:canonical: src.configs.policies.gihh.GIHHConfig.epsilon
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.epsilon
```

````

````{py:attribute} epsilon_decay
:canonical: src.configs.policies.gihh.GIHHConfig.epsilon_decay
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.epsilon_decay
```

````

````{py:attribute} min_epsilon
:canonical: src.configs.policies.gihh.GIHHConfig.min_epsilon
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.min_epsilon
```

````

````{py:attribute} accept_equal
:canonical: src.configs.policies.gihh.GIHHConfig.accept_equal
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.accept_equal
```

````

````{py:attribute} accept_worse_prob
:canonical: src.configs.policies.gihh.GIHHConfig.accept_worse_prob
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.accept_worse_prob
```

````

````{py:attribute} acceptance_decay
:canonical: src.configs.policies.gihh.GIHHConfig.acceptance_decay
:type: float
:value: >
   0.99

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.acceptance_decay
```

````

````{py:attribute} iri_window
:canonical: src.configs.policies.gihh.GIHHConfig.iri_window
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.iri_window
```

````

````{py:attribute} tbi_window
:canonical: src.configs.policies.gihh.GIHHConfig.tbi_window
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.tbi_window
```

````

````{py:attribute} restarts
:canonical: src.configs.policies.gihh.GIHHConfig.restarts
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.restarts
```

````

````{py:attribute} restart_threshold
:canonical: src.configs.policies.gihh.GIHHConfig.restart_threshold
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.restart_threshold
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.gihh.GIHHConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.gihh.GIHHConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.gihh.GIHHConfig.post_processing
```

````

`````
