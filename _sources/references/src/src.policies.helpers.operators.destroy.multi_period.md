# {py:mod}`src.policies.helpers.operators.destroy.multi_period`

```{py:module} src.policies.helpers.operators.destroy.multi_period
```

```{autodoc2-docstring} src.policies.helpers.operators.destroy.multi_period
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`shift_visit_removal <src.policies.helpers.operators.destroy.multi_period.shift_visit_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.multi_period.shift_visit_removal
    :summary:
    ```
* - {py:obj}`pattern_removal <src.policies.helpers.operators.destroy.multi_period.pattern_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.multi_period.pattern_removal
    :summary:
    ```
````

### API

````{py:function} shift_visit_removal(horizon_routes: typing.List[typing.List[typing.List[int]]], n_remove: int, direction: str = 'both', wastes: typing.Optional[typing.Dict[int, float]] = None, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], typing.List[typing.Tuple[int, int]]]
:canonical: src.policies.helpers.operators.destroy.multi_period.shift_visit_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy.multi_period.shift_visit_removal
```
````

````{py:function} pattern_removal(horizon_routes: typing.List[typing.List[typing.List[int]]], n_remove: int, wastes: typing.Optional[typing.Dict[int, float]] = None, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], typing.List[typing.Tuple[int, int]]]
:canonical: src.policies.helpers.operators.destroy.multi_period.pattern_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy.multi_period.pattern_removal
```
````
