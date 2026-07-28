# {py:mod}`src.policies.mandatory_selection.selection_fractional_knapsack`

```{py:module} src.policies.mandatory_selection.selection_fractional_knapsack
```

```{autodoc2-docstring} src.policies.mandatory_selection.selection_fractional_knapsack
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FractionalKnapsackSelection <src.policies.mandatory_selection.selection_fractional_knapsack.FractionalKnapsackSelection>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_fractional_knapsack.FractionalKnapsackSelection
    :summary:
    ```
````

### API

`````{py:class} FractionalKnapsackSelection
:canonical: src.policies.mandatory_selection.selection_fractional_knapsack.FractionalKnapsackSelection

Bases: {py:obj}`logic.src.interfaces.mandatory_selection.IMandatorySelectionStrategy`

```{autodoc2-docstring} src.policies.mandatory_selection.selection_fractional_knapsack.FractionalKnapsackSelection
```

````{py:attribute} _EPS
:canonical: src.policies.mandatory_selection.selection_fractional_knapsack.FractionalKnapsackSelection._EPS
:value: >
   1e-09

```{autodoc2-docstring} src.policies.mandatory_selection.selection_fractional_knapsack.FractionalKnapsackSelection._EPS
```

````

````{py:method} _insertion_distances(candidates: numpy.ndarray, packed: typing.List[int], dm: numpy.ndarray, dist_depot: numpy.ndarray) -> numpy.ndarray
:canonical: src.policies.mandatory_selection.selection_fractional_knapsack.FractionalKnapsackSelection._insertion_distances

```{autodoc2-docstring} src.policies.mandatory_selection.selection_fractional_knapsack.FractionalKnapsackSelection._insertion_distances
```

````

````{py:method} _pack_one_knapsack(eligible: typing.Set[int], revenue: numpy.ndarray, mass: numpy.ndarray, dm: numpy.ndarray, dist_depot: numpy.ndarray, cost_per_km: float, capacity: float) -> typing.List[int]
:canonical: src.policies.mandatory_selection.selection_fractional_knapsack.FractionalKnapsackSelection._pack_one_knapsack

```{autodoc2-docstring} src.policies.mandatory_selection.selection_fractional_knapsack.FractionalKnapsackSelection._pack_one_knapsack
```

````

````{py:method} select_bins(context: logic.src.interfaces.context.SelectionContext) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.SearchContext]
:canonical: src.policies.mandatory_selection.selection_fractional_knapsack.FractionalKnapsackSelection.select_bins

```{autodoc2-docstring} src.policies.mandatory_selection.selection_fractional_knapsack.FractionalKnapsackSelection.select_bins
```

````

`````
