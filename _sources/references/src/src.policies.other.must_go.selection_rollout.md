# {py:mod}`src.policies.other.must_go.selection_rollout`

```{py:module} src.policies.other.must_go.selection_rollout
```

```{autodoc2-docstring} src.policies.other.must_go.selection_rollout
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RolloutSelection <src.policies.other.must_go.selection_rollout.RolloutSelection>`
  - ```{autodoc2-docstring} src.policies.other.must_go.selection_rollout.RolloutSelection
    :summary:
    ```
````

### API

`````{py:class} RolloutSelection
:canonical: src.policies.other.must_go.selection_rollout.RolloutSelection

Bases: {py:obj}`logic.src.interfaces.must_go.IMustGoSelectionStrategy`

```{autodoc2-docstring} src.policies.other.must_go.selection_rollout.RolloutSelection
```

````{py:method} select_bins(context: logic.src.policies.other.must_go.base.selection_context.SelectionContext) -> typing.List[int]
:canonical: src.policies.other.must_go.selection_rollout.RolloutSelection.select_bins

```{autodoc2-docstring} src.policies.other.must_go.selection_rollout.RolloutSelection.select_bins
```

````

````{py:method} _eval_bin(idx: int, collect_today: bool, context: logic.src.policies.other.must_go.base.selection_context.SelectionContext, base_policy: logic.src.interfaces.must_go.IMustGoSelectionStrategy, horizon: int, n_scenarios: int, bin_cap: float, revenue_kg: float, round_trip_cost: numpy.ndarray, max_fill: float, target_sigma: float) -> float
:canonical: src.policies.other.must_go.selection_rollout.RolloutSelection._eval_bin

```{autodoc2-docstring} src.policies.other.must_go.selection_rollout.RolloutSelection._eval_bin
```

````

`````
