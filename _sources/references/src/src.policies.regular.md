# {py:mod}`src.policies.regular`

```{py:module} src.policies.regular
```

```{autodoc2-docstring} src.policies.regular
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RegularPolicy <src.policies.regular.RegularPolicy>`
  - ```{autodoc2-docstring} src.policies.regular.RegularPolicy
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`policy_regular <src.policies.regular.policy_regular>`
  - ```{autodoc2-docstring} src.policies.regular.policy_regular
    :summary:
    ```
````

### API

````{py:function} policy_regular(n_bins: int, bins_waste: numpy.typing.NDArray[numpy.float64], distancesC: numpy.typing.NDArray[numpy.int32], lvl: int, day: int, cached: typing.Optional[typing.List[int]] = None, waste_type: str = 'plastic', area: str = 'riomaior', n_vehicles: int = 1, coords: pandas.DataFrame = None)
:canonical: src.policies.regular.policy_regular

```{autodoc2-docstring} src.policies.regular.policy_regular
```
````

`````{py:class} RegularPolicy
:canonical: src.policies.regular.RegularPolicy

Bases: {py:obj}`src.policies.adapters.IPolicy`

```{autodoc2-docstring} src.policies.regular.RegularPolicy
```

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Any]
:canonical: src.policies.regular.RegularPolicy.execute

```{autodoc2-docstring} src.policies.regular.RegularPolicy.execute
```

````

`````
