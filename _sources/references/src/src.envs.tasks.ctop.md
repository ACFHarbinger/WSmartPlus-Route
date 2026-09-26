# {py:mod}`src.envs.tasks.ctop`

```{py:module} src.envs.tasks.ctop
```

```{autodoc2-docstring} src.envs.tasks.ctop
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CTOP <src.envs.tasks.ctop.CTOP>`
  - ```{autodoc2-docstring} src.envs.tasks.ctop.CTOP
    :summary:
    ```
````

### API

`````{py:class} CTOP
:canonical: src.envs.tasks.ctop.CTOP

Bases: {py:obj}`logic.src.envs.tasks.cvrpp.CVRPP`

```{autodoc2-docstring} src.envs.tasks.ctop.CTOP
```

````{py:attribute} NAME
:canonical: src.envs.tasks.ctop.CTOP.NAME
:value: >
   'ctop'

```{autodoc2-docstring} src.envs.tasks.ctop.CTOP.NAME
```

````

````{py:method} get_costs(dataset: typing.Dict[str, typing.Any], pi: torch.Tensor, cw_dict: typing.Optional[typing.Dict[str, float]], dist_matrix: typing.Optional[torch.Tensor] = None) -> typing.Tuple[torch.Tensor, typing.Dict[str, torch.Tensor], None]
:canonical: src.envs.tasks.ctop.CTOP.get_costs
:staticmethod:

```{autodoc2-docstring} src.envs.tasks.ctop.CTOP.get_costs
```

````

`````
