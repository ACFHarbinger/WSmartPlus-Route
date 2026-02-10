# {py:mod}`src.policies.other.must_go.selection_lookahead`

```{py:module} src.policies.other.must_go.selection_lookahead
```

```{autodoc2-docstring} src.policies.other.must_go.selection_lookahead
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LookaheadSelection <src.policies.other.must_go.selection_lookahead.LookaheadSelection>`
  - ```{autodoc2-docstring} src.policies.other.must_go.selection_lookahead.LookaheadSelection
    :summary:
    ```
````

### API

`````{py:class} LookaheadSelection
:canonical: src.policies.other.must_go.selection_lookahead.LookaheadSelection

Bases: {py:obj}`logic.src.interfaces.must_go.IMustGoSelectionStrategy`

```{autodoc2-docstring} src.policies.other.must_go.selection_lookahead.LookaheadSelection
```

````{py:method} _should_bin_be_collected(current_fill_level: float, accumulation_rate: float) -> bool
:canonical: src.policies.other.must_go.selection_lookahead.LookaheadSelection._should_bin_be_collected

```{autodoc2-docstring} src.policies.other.must_go.selection_lookahead.LookaheadSelection._should_bin_be_collected
```

````

````{py:method} _update_fill_levels_after_first_collection(bin_indices: typing.List[int], must_go_bins: typing.List[int], current_fill_levels: numpy.ndarray) -> numpy.ndarray
:canonical: src.policies.other.must_go.selection_lookahead.LookaheadSelection._update_fill_levels_after_first_collection

```{autodoc2-docstring} src.policies.other.must_go.selection_lookahead.LookaheadSelection._update_fill_levels_after_first_collection
```

````

````{py:method} _initialize_lists_bins(n_bins: int) -> typing.List[int]
:canonical: src.policies.other.must_go.selection_lookahead.LookaheadSelection._initialize_lists_bins

```{autodoc2-docstring} src.policies.other.must_go.selection_lookahead.LookaheadSelection._initialize_lists_bins
```

````

````{py:method} _calculate_next_collection_days(bin_indices: typing.List[int], must_go_bins: typing.List[int], current_fill_levels: numpy.ndarray, accumulation_rates: numpy.ndarray) -> typing.List[int]
:canonical: src.policies.other.must_go.selection_lookahead.LookaheadSelection._calculate_next_collection_days

```{autodoc2-docstring} src.policies.other.must_go.selection_lookahead.LookaheadSelection._calculate_next_collection_days
```

````

````{py:method} _get_next_collection_day(bin_indices: typing.List[int], must_go_bins: typing.List[int], current_fill_levels: numpy.ndarray, accumulation_rates: numpy.ndarray) -> int
:canonical: src.policies.other.must_go.selection_lookahead.LookaheadSelection._get_next_collection_day

```{autodoc2-docstring} src.policies.other.must_go.selection_lookahead.LookaheadSelection._get_next_collection_day
```

````

````{py:method} _add_bins_to_collect(bin_indices: typing.List[int], next_collection_day: int, must_go_bins: typing.List[int], current_fill_levels: numpy.ndarray, accumulation_rates: numpy.ndarray) -> typing.List[int]
:canonical: src.policies.other.must_go.selection_lookahead.LookaheadSelection._add_bins_to_collect

```{autodoc2-docstring} src.policies.other.must_go.selection_lookahead.LookaheadSelection._add_bins_to_collect
```

````

````{py:method} select_bins(context: logic.src.policies.other.must_go.base.selection_context.SelectionContext) -> typing.List[int]
:canonical: src.policies.other.must_go.selection_lookahead.LookaheadSelection.select_bins

```{autodoc2-docstring} src.policies.other.must_go.selection_lookahead.LookaheadSelection.select_bins
```

````

`````
