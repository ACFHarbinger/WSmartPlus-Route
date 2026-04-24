# {py:mod}`src.policies.mandatory_selection.selection_fptas_knapsack`

```{py:module} src.policies.mandatory_selection.selection_fptas_knapsack
```

```{autodoc2-docstring} src.policies.mandatory_selection.selection_fptas_knapsack
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FPTASKnapsackSelection <src.policies.mandatory_selection.selection_fptas_knapsack.FPTASKnapsackSelection>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_fptas_knapsack.FPTASKnapsackSelection
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_compute_overflow_risk <src.policies.mandatory_selection.selection_fptas_knapsack._compute_overflow_risk>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_fptas_knapsack._compute_overflow_risk
    :summary:
    ```
* - {py:obj}`_fptas_01_knapsack <src.policies.mandatory_selection.selection_fptas_knapsack._fptas_01_knapsack>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_fptas_knapsack._fptas_01_knapsack
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_EPS_GUARD <src.policies.mandatory_selection.selection_fptas_knapsack._EPS_GUARD>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_fptas_knapsack._EPS_GUARD
    :summary:
    ```
* - {py:obj}`_DIST_EPS <src.policies.mandatory_selection.selection_fptas_knapsack._DIST_EPS>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_fptas_knapsack._DIST_EPS
    :summary:
    ```
* - {py:obj}`_MAX_DP_VALUES <src.policies.mandatory_selection.selection_fptas_knapsack._MAX_DP_VALUES>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_fptas_knapsack._MAX_DP_VALUES
    :summary:
    ```
````

### API

````{py:data} _EPS_GUARD
:canonical: src.policies.mandatory_selection.selection_fptas_knapsack._EPS_GUARD
:type: float
:value: >
   1e-09

```{autodoc2-docstring} src.policies.mandatory_selection.selection_fptas_knapsack._EPS_GUARD
```

````

````{py:data} _DIST_EPS
:canonical: src.policies.mandatory_selection.selection_fptas_knapsack._DIST_EPS
:type: float
:value: >
   1e-09

```{autodoc2-docstring} src.policies.mandatory_selection.selection_fptas_knapsack._DIST_EPS
```

````

````{py:data} _MAX_DP_VALUES
:canonical: src.policies.mandatory_selection.selection_fptas_knapsack._MAX_DP_VALUES
:type: int
:value: >
   100000

```{autodoc2-docstring} src.policies.mandatory_selection.selection_fptas_knapsack._MAX_DP_VALUES
```

````

````{py:function} _compute_overflow_risk(current_fill: numpy.ndarray, bin_mass: numpy.ndarray, scenario_tree: typing.Optional[typing.Any], overflow_penalty_frac: float) -> numpy.ndarray
:canonical: src.policies.mandatory_selection.selection_fptas_knapsack._compute_overflow_risk

```{autodoc2-docstring} src.policies.mandatory_selection.selection_fptas_knapsack._compute_overflow_risk
```
````

````{py:function} _fptas_01_knapsack(values: numpy.ndarray, weights: numpy.ndarray, capacity: float, epsilon: float) -> numpy.ndarray
:canonical: src.policies.mandatory_selection.selection_fptas_knapsack._fptas_01_knapsack

```{autodoc2-docstring} src.policies.mandatory_selection.selection_fptas_knapsack._fptas_01_knapsack
```
````

`````{py:class} FPTASKnapsackSelection(epsilon: float = 0.1, alpha: float = 0.5, overflow_penalty_frac: float = 1.0)
:canonical: src.policies.mandatory_selection.selection_fptas_knapsack.FPTASKnapsackSelection

Bases: {py:obj}`logic.src.interfaces.mandatory_selection.IMandatorySelectionStrategy`

```{autodoc2-docstring} src.policies.mandatory_selection.selection_fptas_knapsack.FPTASKnapsackSelection
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.mandatory_selection.selection_fptas_knapsack.FPTASKnapsackSelection.__init__
```

````{py:method} select_bins(context: logic.src.interfaces.context.SelectionContext) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.SearchContext]
:canonical: src.policies.mandatory_selection.selection_fptas_knapsack.FPTASKnapsackSelection.select_bins

```{autodoc2-docstring} src.policies.mandatory_selection.selection_fptas_knapsack.FPTASKnapsackSelection.select_bins
```

````

`````
