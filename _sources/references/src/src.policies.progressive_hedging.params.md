# {py:mod}`src.policies.progressive_hedging.params`

```{py:module} src.policies.progressive_hedging.params
```

```{autodoc2-docstring} src.policies.progressive_hedging.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PHParams <src.policies.progressive_hedging.params.PHParams>`
  - ```{autodoc2-docstring} src.policies.progressive_hedging.params.PHParams
    :summary:
    ```
````

### API

`````{py:class} PHParams
:canonical: src.policies.progressive_hedging.params.PHParams

```{autodoc2-docstring} src.policies.progressive_hedging.params.PHParams
```

````{py:attribute} rho
:canonical: src.policies.progressive_hedging.params.PHParams.rho
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.progressive_hedging.params.PHParams.rho
```

````

````{py:attribute} max_iterations
:canonical: src.policies.progressive_hedging.params.PHParams.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.progressive_hedging.params.PHParams.max_iterations
```

````

````{py:attribute} convergence_tol
:canonical: src.policies.progressive_hedging.params.PHParams.convergence_tol
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.progressive_hedging.params.PHParams.convergence_tol
```

````

````{py:attribute} sub_solver
:canonical: src.policies.progressive_hedging.params.PHParams.sub_solver
:type: str
:value: >
   'bc'

```{autodoc2-docstring} src.policies.progressive_hedging.params.PHParams.sub_solver
```

````

````{py:attribute} num_scenarios
:canonical: src.policies.progressive_hedging.params.PHParams.num_scenarios
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.progressive_hedging.params.PHParams.num_scenarios
```

````

````{py:attribute} time_limit
:canonical: src.policies.progressive_hedging.params.PHParams.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.policies.progressive_hedging.params.PHParams.time_limit
```

````

````{py:attribute} verbose
:canonical: src.policies.progressive_hedging.params.PHParams.verbose
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.progressive_hedging.params.PHParams.verbose
```

````

````{py:attribute} seed
:canonical: src.policies.progressive_hedging.params.PHParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.progressive_hedging.params.PHParams.seed
```

````

````{py:method} from_config(config: typing.Dict[str, typing.Any]) -> src.policies.progressive_hedging.params.PHParams
:canonical: src.policies.progressive_hedging.params.PHParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.progressive_hedging.params.PHParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.progressive_hedging.params.PHParams.to_dict

```{autodoc2-docstring} src.policies.progressive_hedging.params.PHParams.to_dict
```

````

`````
