# {py:mod}`src.policies.genetic_programming_hyper_heuristic.params`

```{py:module} src.policies.genetic_programming_hyper_heuristic.params
```

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GPHHParams <src.policies.genetic_programming_hyper_heuristic.params.GPHHParams>`
  - ```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.params.GPHHParams
    :summary:
    ```
````

### API

`````{py:class} GPHHParams
:canonical: src.policies.genetic_programming_hyper_heuristic.params.GPHHParams

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.params.GPHHParams
```

````{py:attribute} gp_pop_size
:canonical: src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.gp_pop_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.gp_pop_size
```

````

````{py:attribute} max_gp_generations
:canonical: src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.max_gp_generations
:type: int
:value: >
   30

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.max_gp_generations
```

````

````{py:attribute} tree_depth
:canonical: src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.tree_depth
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.tree_depth
```

````

````{py:attribute} tournament_size
:canonical: src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.tournament_size
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.tournament_size
```

````

````{py:attribute} time_limit
:canonical: src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.time_limit
```

````

````{py:attribute} parsimony_coefficient
:canonical: src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.parsimony_coefficient
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.parsimony_coefficient
```

````

````{py:attribute} candidate_list_size
:canonical: src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.candidate_list_size
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.candidate_list_size
```

````

````{py:attribute} n_training_instances
:canonical: src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.n_training_instances
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.n_training_instances
```

````

````{py:attribute} training_sample_ratio
:canonical: src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.training_sample_ratio
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.training_sample_ratio
```

````

````{py:attribute} seed
:canonical: src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.vrpp
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.genetic_programming_hyper_heuristic.params.GPHHParams
:canonical: src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.params.GPHHParams.from_config
```

````

`````
