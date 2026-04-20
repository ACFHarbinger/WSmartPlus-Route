# {py:mod}`src.policies.mandatory_selection.selection_rollout`

```{py:module} src.policies.mandatory_selection.selection_rollout
```

```{autodoc2-docstring} src.policies.mandatory_selection.selection_rollout
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RolloutSelection <src.policies.mandatory_selection.selection_rollout.RolloutSelection>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_rollout.RolloutSelection
    :summary:
    ```
````

### API

`````{py:class} RolloutSelection
:canonical: src.policies.mandatory_selection.selection_rollout.RolloutSelection

Bases: {py:obj}`logic.src.interfaces.mandatory_selection.IMandatorySelectionStrategy`

```{autodoc2-docstring} src.policies.mandatory_selection.selection_rollout.RolloutSelection
```

````{py:method} select_bins(context: logic.src.policies.mandatory_selection.base.selection_context.SelectionContext) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.search_context.SearchContext]
:canonical: src.policies.mandatory_selection.selection_rollout.RolloutSelection.select_bins

```{autodoc2-docstring} src.policies.mandatory_selection.selection_rollout.RolloutSelection.select_bins
```

````

````{py:method} _eval_bin(idx: int, collect_today: bool, context: logic.src.policies.mandatory_selection.base.selection_context.SelectionContext, base_policy: logic.src.interfaces.mandatory_selection.IMandatorySelectionStrategy, horizon: int, n_scenarios: int, bin_cap: float, revenue_kg: float, round_trip_cost: numpy.ndarray, max_fill: float, target_sigma: float) -> float
:canonical: src.policies.mandatory_selection.selection_rollout.RolloutSelection._eval_bin

```{autodoc2-docstring} src.policies.mandatory_selection.selection_rollout.RolloutSelection._eval_bin
```

````

`````
