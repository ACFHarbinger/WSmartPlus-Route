# {py:mod}`src.policies.look_ahead_aux.update`

```{py:module} src.policies.look_ahead_aux.update
```

```{autodoc2-docstring} src.policies.look_ahead_aux.update
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`should_bin_be_collected <src.policies.look_ahead_aux.update.should_bin_be_collected>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.update.should_bin_be_collected
    :summary:
    ```
* - {py:obj}`add_bins_to_collect <src.policies.look_ahead_aux.update.add_bins_to_collect>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.update.add_bins_to_collect
    :summary:
    ```
* - {py:obj}`update_fill_levels_after_first_collection <src.policies.look_ahead_aux.update.update_fill_levels_after_first_collection>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.update.update_fill_levels_after_first_collection
    :summary:
    ```
* - {py:obj}`initialize_lists_of_bins <src.policies.look_ahead_aux.update.initialize_lists_of_bins>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.update.initialize_lists_of_bins
    :summary:
    ```
* - {py:obj}`calculate_next_collection_days <src.policies.look_ahead_aux.update.calculate_next_collection_days>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.update.calculate_next_collection_days
    :summary:
    ```
* - {py:obj}`get_next_collection_day <src.policies.look_ahead_aux.update.get_next_collection_day>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.update.get_next_collection_day
    :summary:
    ```
````

### API

````{py:function} should_bin_be_collected(current_fill_level, accumulation_rate)
:canonical: src.policies.look_ahead_aux.update.should_bin_be_collected

```{autodoc2-docstring} src.policies.look_ahead_aux.update.should_bin_be_collected
```
````

````{py:function} add_bins_to_collect(binsids, next_collection_day, must_go_bins, current_fill_levels, accumulation_rates, current_collection_day)
:canonical: src.policies.look_ahead_aux.update.add_bins_to_collect

```{autodoc2-docstring} src.policies.look_ahead_aux.update.add_bins_to_collect
```
````

````{py:function} update_fill_levels_after_first_collection(binsids, must_go_bins, current_fill_levels)
:canonical: src.policies.look_ahead_aux.update.update_fill_levels_after_first_collection

```{autodoc2-docstring} src.policies.look_ahead_aux.update.update_fill_levels_after_first_collection
```
````

````{py:function} initialize_lists_of_bins(binsids)
:canonical: src.policies.look_ahead_aux.update.initialize_lists_of_bins

```{autodoc2-docstring} src.policies.look_ahead_aux.update.initialize_lists_of_bins
```
````

````{py:function} calculate_next_collection_days(must_go_bins, current_fill_levels, accumulation_rates, binsids)
:canonical: src.policies.look_ahead_aux.update.calculate_next_collection_days

```{autodoc2-docstring} src.policies.look_ahead_aux.update.calculate_next_collection_days
```
````

````{py:function} get_next_collection_day(must_go_bins, current_fill_levels, accumulation_rates, binsids)
:canonical: src.policies.look_ahead_aux.update.get_next_collection_day

```{autodoc2-docstring} src.policies.look_ahead_aux.update.get_next_collection_day
```
````
