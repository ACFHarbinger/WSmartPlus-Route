# {py:mod}`src.configs.policies.ts`

```{py:module} src.configs.policies.ts
```

```{autodoc2-docstring} src.configs.policies.ts
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TSConfig <src.configs.policies.ts.TSConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ts.TSConfig
    :summary:
    ```
````

### API

`````{py:class} TSConfig
:canonical: src.configs.policies.ts.TSConfig

```{autodoc2-docstring} src.configs.policies.ts.TSConfig
```

````{py:attribute} tabu_tenure
:canonical: src.configs.policies.ts.TSConfig.tabu_tenure
:type: int
:value: >
   7

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.tabu_tenure
```

````

````{py:attribute} dynamic_tenure
:canonical: src.configs.policies.ts.TSConfig.dynamic_tenure
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.dynamic_tenure
```

````

````{py:attribute} min_tenure
:canonical: src.configs.policies.ts.TSConfig.min_tenure
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.min_tenure
```

````

````{py:attribute} max_tenure
:canonical: src.configs.policies.ts.TSConfig.max_tenure
:type: int
:value: >
   15

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.max_tenure
```

````

````{py:attribute} aspiration_enabled
:canonical: src.configs.policies.ts.TSConfig.aspiration_enabled
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.aspiration_enabled
```

````

````{py:attribute} intensification_enabled
:canonical: src.configs.policies.ts.TSConfig.intensification_enabled
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.intensification_enabled
```

````

````{py:attribute} diversification_enabled
:canonical: src.configs.policies.ts.TSConfig.diversification_enabled
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.diversification_enabled
```

````

````{py:attribute} intensification_interval
:canonical: src.configs.policies.ts.TSConfig.intensification_interval
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.intensification_interval
```

````

````{py:attribute} diversification_interval
:canonical: src.configs.policies.ts.TSConfig.diversification_interval
:type: int
:value: >
   200

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.diversification_interval
```

````

````{py:attribute} elite_size
:canonical: src.configs.policies.ts.TSConfig.elite_size
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.elite_size
```

````

````{py:attribute} frequency_penalty_weight
:canonical: src.configs.policies.ts.TSConfig.frequency_penalty_weight
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.frequency_penalty_weight
```

````

````{py:attribute} candidate_list_enabled
:canonical: src.configs.policies.ts.TSConfig.candidate_list_enabled
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.candidate_list_enabled
```

````

````{py:attribute} candidate_list_size
:canonical: src.configs.policies.ts.TSConfig.candidate_list_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.candidate_list_size
```

````

````{py:attribute} oscillation_enabled
:canonical: src.configs.policies.ts.TSConfig.oscillation_enabled
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.oscillation_enabled
```

````

````{py:attribute} feasibility_tolerance
:canonical: src.configs.policies.ts.TSConfig.feasibility_tolerance
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.feasibility_tolerance
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.ts.TSConfig.max_iterations
:type: int
:value: >
   5000

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.max_iterations
```

````

````{py:attribute} max_iterations_no_improve
:canonical: src.configs.policies.ts.TSConfig.max_iterations_no_improve
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.max_iterations_no_improve
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.ts.TSConfig.n_removal
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.ts.TSConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.ts.TSConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.time_limit
```

````

````{py:attribute} use_swap
:canonical: src.configs.policies.ts.TSConfig.use_swap
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.use_swap
```

````

````{py:attribute} use_relocate
:canonical: src.configs.policies.ts.TSConfig.use_relocate
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.use_relocate
```

````

````{py:attribute} use_2opt
:canonical: src.configs.policies.ts.TSConfig.use_2opt
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.use_2opt
```

````

````{py:attribute} use_insertion
:canonical: src.configs.policies.ts.TSConfig.use_insertion
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.use_insertion
```

````

````{py:attribute} seed
:canonical: src.configs.policies.ts.TSConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.ts.TSConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.ts.TSConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.profit_aware_operators
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.ts.TSConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.ts.TSConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ts.TSConfig.post_processing
```

````

`````
