# {py:mod}`src.policies.mandatory_selection.selection_mip_knapsack`

```{py:module} src.policies.mandatory_selection.selection_mip_knapsack
```

```{autodoc2-docstring} src.policies.mandatory_selection.selection_mip_knapsack
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MIPKnapsackSelection <src.policies.mandatory_selection.selection_mip_knapsack.MIPKnapsackSelection>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_mip_knapsack.MIPKnapsackSelection
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_compute_overflow_risk <src.policies.mandatory_selection.selection_mip_knapsack._compute_overflow_risk>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_mip_knapsack._compute_overflow_risk
    :summary:
    ```
````

### API

````{py:function} _compute_overflow_risk(current_fill: numpy.ndarray, bin_mass: numpy.ndarray, scenario_tree: typing.Optional[typing.Any], overflow_penalty_frac: float) -> numpy.ndarray
:canonical: src.policies.mandatory_selection.selection_mip_knapsack._compute_overflow_risk

```{autodoc2-docstring} src.policies.mandatory_selection.selection_mip_knapsack._compute_overflow_risk
```
````

`````{py:class} MIPKnapsackSelection(overflow_penalty_frac: float = 1.0)
:canonical: src.policies.mandatory_selection.selection_mip_knapsack.MIPKnapsackSelection

Bases: {py:obj}`logic.src.interfaces.mandatory_selection.IMandatorySelectionStrategy`

```{autodoc2-docstring} src.policies.mandatory_selection.selection_mip_knapsack.MIPKnapsackSelection
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.mandatory_selection.selection_mip_knapsack.MIPKnapsackSelection.__init__
```

````{py:method} select_bins(context: logic.src.interfaces.context.SelectionContext) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.SearchContext]
:canonical: src.policies.mandatory_selection.selection_mip_knapsack.MIPKnapsackSelection.select_bins

```{autodoc2-docstring} src.policies.mandatory_selection.selection_mip_knapsack.MIPKnapsackSelection.select_bins
```

````

`````
