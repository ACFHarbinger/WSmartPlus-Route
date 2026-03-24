# {py:mod}`src.policies.fast_iterative_localized_optimization.ruin_recreate`

```{py:module} src.policies.fast_iterative_localized_optimization.ruin_recreate
```

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.ruin_recreate
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RuinAndRecreate <src.policies.fast_iterative_localized_optimization.ruin_recreate.RuinAndRecreate>`
  - ```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.ruin_recreate.RuinAndRecreate
    :summary:
    ```
````

### API

`````{py:class} RuinAndRecreate(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, rng: numpy.random.Generator, profit_aware_operators: bool = False, vrpp: bool = True)
:canonical: src.policies.fast_iterative_localized_optimization.ruin_recreate.RuinAndRecreate

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.ruin_recreate.RuinAndRecreate
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.ruin_recreate.RuinAndRecreate.__init__
```

````{py:method} apply(routes: typing.List[typing.List[int]], omega: typing.List[int], all_customers: typing.List[int], mandatory_nodes: typing.List[int]) -> typing.Tuple[typing.List[typing.List[int]], int, typing.List[int]]
:canonical: src.policies.fast_iterative_localized_optimization.ruin_recreate.RuinAndRecreate.apply

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.ruin_recreate.RuinAndRecreate.apply
```

````

`````
