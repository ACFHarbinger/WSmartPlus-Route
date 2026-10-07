# {py:mod}`src.envs.tasks.tcmvptp`

```{py:module} src.envs.tasks.tcmvptp
```

```{autodoc2-docstring} src.envs.tasks.tcmvptp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TCMVPTP <src.envs.tasks.tcmvptp.TCMVPTP>`
  - ```{autodoc2-docstring} src.envs.tasks.tcmvptp.TCMVPTP
    :summary:
    ```
````

### API

`````{py:class} TCMVPTP
:canonical: src.envs.tasks.tcmvptp.TCMVPTP

Bases: {py:obj}`logic.src.envs.tasks.mvptp.MVPTP`

```{autodoc2-docstring} src.envs.tasks.tcmvptp.TCMVPTP
```

````{py:attribute} NAME
:canonical: src.envs.tasks.tcmvptp.TCMVPTP.NAME
:value: >
   'tcmvptp'

```{autodoc2-docstring} src.envs.tasks.tcmvptp.TCMVPTP.NAME
```

````

````{py:method} get_costs(dataset: typing.Dict[str, typing.Any], pi: torch.Tensor, cw_dict: typing.Optional[typing.Dict[str, float]], dist_matrix: typing.Optional[torch.Tensor] = None) -> typing.Tuple[torch.Tensor, typing.Dict[str, torch.Tensor], None]
:canonical: src.envs.tasks.tcmvptp.TCMVPTP.get_costs
:staticmethod:

```{autodoc2-docstring} src.envs.tasks.tcmvptp.TCMVPTP.get_costs
```

````

`````
