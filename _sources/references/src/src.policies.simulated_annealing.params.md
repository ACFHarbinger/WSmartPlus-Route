# {py:mod}`src.policies.simulated_annealing.params`

```{py:module} src.policies.simulated_annealing.params
```

```{autodoc2-docstring} src.policies.simulated_annealing.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SAParams <src.policies.simulated_annealing.params.SAParams>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing.params.SAParams
    :summary:
    ```
````

### API

`````{py:class} SAParams
:canonical: src.policies.simulated_annealing.params.SAParams

```{autodoc2-docstring} src.policies.simulated_annealing.params.SAParams
```

````{py:attribute} initial_temp
:canonical: src.policies.simulated_annealing.params.SAParams.initial_temp
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.policies.simulated_annealing.params.SAParams.initial_temp
```

````

````{py:attribute} alpha
:canonical: src.policies.simulated_annealing.params.SAParams.alpha
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.policies.simulated_annealing.params.SAParams.alpha
```

````

````{py:attribute} min_temp
:canonical: src.policies.simulated_annealing.params.SAParams.min_temp
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.simulated_annealing.params.SAParams.min_temp
```

````

````{py:attribute} max_iterations
:canonical: src.policies.simulated_annealing.params.SAParams.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.policies.simulated_annealing.params.SAParams.max_iterations
```

````

````{py:attribute} n_removal
:canonical: src.policies.simulated_annealing.params.SAParams.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.simulated_annealing.params.SAParams.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.policies.simulated_annealing.params.SAParams.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.simulated_annealing.params.SAParams.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.policies.simulated_annealing.params.SAParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.simulated_annealing.params.SAParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.simulated_annealing.params.SAParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.simulated_annealing.params.SAParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.simulated_annealing.params.SAParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.simulated_annealing.params.SAParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.simulated_annealing.params.SAParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.simulated_annealing.params.SAParams.profit_aware_operators
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.simulated_annealing.params.SAParams
:canonical: src.policies.simulated_annealing.params.SAParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.simulated_annealing.params.SAParams.from_config
```

````

`````
