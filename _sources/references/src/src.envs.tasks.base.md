# {py:mod}`src.envs.tasks.base`

```{py:module} src.envs.tasks.base
```

```{autodoc2-docstring} src.envs.tasks.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BaseProblem <src.envs.tasks.base.BaseProblem>`
  - ```{autodoc2-docstring} src.envs.tasks.base.BaseProblem
    :summary:
    ```
````

### API

`````{py:class} BaseProblem
:canonical: src.envs.tasks.base.BaseProblem

```{autodoc2-docstring} src.envs.tasks.base.BaseProblem
```

````{py:attribute} NAME
:canonical: src.envs.tasks.base.BaseProblem.NAME
:type: str
:value: >
   'base'

```{autodoc2-docstring} src.envs.tasks.base.BaseProblem.NAME
```

````

````{py:method} validate_tours(pi: torch.Tensor) -> bool
:canonical: src.envs.tasks.base.BaseProblem.validate_tours
:staticmethod:

```{autodoc2-docstring} src.envs.tasks.base.BaseProblem.validate_tours
```

````

````{py:method} get_tour_length(dataset: typing.Dict[str, typing.Any], pi: torch.Tensor, dist_matrix: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.tasks.base.BaseProblem.get_tour_length
:staticmethod:

```{autodoc2-docstring} src.envs.tasks.base.BaseProblem.get_tour_length
```

````

````{py:method} beam_search(input, beam_size, cost_weights, model=None, **kwargs)
:canonical: src.envs.tasks.base.BaseProblem.beam_search
:classmethod:

```{autodoc2-docstring} src.envs.tasks.base.BaseProblem.beam_search
```

````

````{py:method} make_state(input_data: typing.Any, edges: typing.Any = None, cost_weights: typing.Any = None, dist_matrix: typing.Any = None, **kwargs: typing.Any) -> typing.Any
:canonical: src.envs.tasks.base.BaseProblem.make_state
:classmethod:

```{autodoc2-docstring} src.envs.tasks.base.BaseProblem.make_state
```

````

`````
