# {py:mod}`src.envs.tasks.thop`

```{py:module} src.envs.tasks.thop
```

```{autodoc2-docstring} src.envs.tasks.thop
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ThOP <src.envs.tasks.thop.ThOP>`
  - ```{autodoc2-docstring} src.envs.tasks.thop.ThOP
    :summary:
    ```
````

### API

`````{py:class} ThOP
:canonical: src.envs.tasks.thop.ThOP

Bases: {py:obj}`logic.src.envs.tasks.base.BaseProblem`

```{autodoc2-docstring} src.envs.tasks.thop.ThOP
```

````{py:attribute} NAME
:canonical: src.envs.tasks.thop.ThOP.NAME
:value: >
   'thop'

```{autodoc2-docstring} src.envs.tasks.thop.ThOP.NAME
```

````

````{py:method} get_costs(dataset: typing.Dict[str, typing.Any], pi: torch.Tensor, cw_dict: typing.Optional[typing.Dict[str, typing.Any]], dist_matrix: typing.Optional[torch.Tensor] = None) -> typing.Tuple[torch.Tensor, typing.Dict[str, typing.Any], None]
:canonical: src.envs.tasks.thop.ThOP.get_costs
:staticmethod:

```{autodoc2-docstring} src.envs.tasks.thop.ThOP.get_costs
```

````

`````
