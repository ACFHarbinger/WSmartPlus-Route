# {py:mod}`src.envs.tasks.tsp`

```{py:module} src.envs.tasks.tsp
```

```{autodoc2-docstring} src.envs.tasks.tsp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TSP <src.envs.tasks.tsp.TSP>`
  - ```{autodoc2-docstring} src.envs.tasks.tsp.TSP
    :summary:
    ```
````

### API

`````{py:class} TSP
:canonical: src.envs.tasks.tsp.TSP

Bases: {py:obj}`logic.src.envs.tasks.base.BaseProblem`

```{autodoc2-docstring} src.envs.tasks.tsp.TSP
```

````{py:attribute} NAME
:canonical: src.envs.tasks.tsp.TSP.NAME
:value: >
   'tsp'

```{autodoc2-docstring} src.envs.tasks.tsp.TSP.NAME
```

````

````{py:method} get_costs(dataset: typing.Dict[str, typing.Any], pi: torch.Tensor, cw_dict: typing.Optional[typing.Dict[str, typing.Any]], dist_matrix: typing.Optional[torch.Tensor] = None) -> typing.Tuple[torch.Tensor, typing.Dict[str, typing.Any], None]
:canonical: src.envs.tasks.tsp.TSP.get_costs
:staticmethod:

```{autodoc2-docstring} src.envs.tasks.tsp.TSP.get_costs
```

````

````{py:method} _cost_from_locs(pi: torch.Tensor, locs: torch.Tensor) -> torch.Tensor
:canonical: src.envs.tasks.tsp.TSP._cost_from_locs
:staticmethod:

```{autodoc2-docstring} src.envs.tasks.tsp.TSP._cost_from_locs
```

````

````{py:method} _cost_from_matrix(pi: torch.Tensor, mat: torch.Tensor) -> torch.Tensor
:canonical: src.envs.tasks.tsp.TSP._cost_from_matrix
:staticmethod:

```{autodoc2-docstring} src.envs.tasks.tsp.TSP._cost_from_matrix
```

````

`````
