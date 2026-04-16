# {py:mod}`src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update`

```{py:module} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`should_bin_be_collected <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.should_bin_be_collected>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.should_bin_be_collected
    :summary:
    ```
* - {py:obj}`add_bins_to_collect <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.add_bins_to_collect>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.add_bins_to_collect
    :summary:
    ```
* - {py:obj}`update_fill_levels_after_first_collection <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.update_fill_levels_after_first_collection>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.update_fill_levels_after_first_collection
    :summary:
    ```
* - {py:obj}`initialize_lists_of_bins <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.initialize_lists_of_bins>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.initialize_lists_of_bins
    :summary:
    ```
* - {py:obj}`calculate_next_collection_days <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.calculate_next_collection_days>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.calculate_next_collection_days
    :summary:
    ```
* - {py:obj}`get_next_collection_day <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.get_next_collection_day>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.get_next_collection_day
    :summary:
    ```
````

### API

````{py:function} should_bin_be_collected(current_fill_level, accumulation_rate)
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.should_bin_be_collected

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.should_bin_be_collected
```
````

````{py:function} add_bins_to_collect(binsids, next_collection_day, mandatory_bins, current_fill_levels, accumulation_rates, current_collection_day)
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.add_bins_to_collect

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.add_bins_to_collect
```
````

````{py:function} update_fill_levels_after_first_collection(binsids, mandatory_bins, current_fill_levels)
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.update_fill_levels_after_first_collection

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.update_fill_levels_after_first_collection
```
````

````{py:function} initialize_lists_of_bins(binsids)
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.initialize_lists_of_bins

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.initialize_lists_of_bins
```
````

````{py:function} calculate_next_collection_days(mandatory_bins, current_fill_levels, accumulation_rates, binsids)
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.calculate_next_collection_days

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.calculate_next_collection_days
```
````

````{py:function} get_next_collection_day(mandatory_bins, current_fill_levels, accumulation_rates, binsids)
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.get_next_collection_day

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.update.get_next_collection_day
```
````
