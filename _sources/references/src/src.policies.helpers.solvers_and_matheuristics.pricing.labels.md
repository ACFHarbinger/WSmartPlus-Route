# {py:mod}`src.policies.helpers.solvers_and_matheuristics.pricing.labels`

```{py:module} src.policies.helpers.solvers_and_matheuristics.pricing.labels
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.labels
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Label <src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label
    :summary:
    ```
````

### API

`````{py:class} Label
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label
```

````{py:attribute} reduced_cost
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.reduced_cost
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.reduced_cost
```

````

````{py:attribute} node
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.node
:type: int
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.node
```

````

````{py:attribute} load
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.load
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.load
```

````

````{py:attribute} visited
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.visited
:type: typing.Set[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.visited
```

````

````{py:attribute} ng_memory
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.ng_memory
:type: typing.Set[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.ng_memory
```

````

````{py:attribute} rf_unmatched
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.rf_unmatched
:type: typing.FrozenSet[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.rf_unmatched
```

````

````{py:attribute} parent
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.parent
:type: typing.Optional[src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.parent
```

````

````{py:attribute} sri_state
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.sri_state
:type: typing.Tuple[int, ...]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.sri_state
```

````

````{py:method} dominates(other: src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label, use_ng: bool = False, epsilon: float = 1e-06, sri_dual_values: typing.Optional[typing.List[float]] = None) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.dominates

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.dominates
```

````

````{py:method} is_feasible(capacity: float) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.is_feasible

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.is_feasible
```

````

````{py:method} reconstruct_path() -> typing.List[int]
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.reconstruct_path

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.labels.Label.reconstruct_path
```

````

`````
