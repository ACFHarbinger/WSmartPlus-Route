# {py:mod}`src.policies`

```{py:module} src.policies
```

```{autodoc2-docstring} src.policies
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

src.policies.vehicle_routing_problem_with_profits
src.policies.operators
src.policies.simulated_annealing_neighborhood_search
src.policies.slack_induction_by_string_removal
src.policies.neural_agent
src.policies.adaptive_large_neighborhood_search
src.policies.ant_colony_optimization
src.policies.hybrid_genetic_search
src.policies.other
src.policies.local_search
src.policies.adapters
src.policies.branch_cut_and_price
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.policies.lin_kernighan_helsgaun
src.policies.cvrp
src.policies.hgs_alns
src.policies.tsp
```

## Package Contents

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.policies.__all__>`
  - ```{autodoc2-docstring} src.policies.__all__
    :summary:
    ```
* - {py:obj}`create_policy <src.policies.create_policy>`
  - ```{autodoc2-docstring} src.policies.create_policy
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: src.policies.__all__
:value: >
   ['ALNSParams', 'PolicyFactory', 'MustGoSelectionFactory', 'MustGoSelectionRegistry', 'SelectionConte...

```{autodoc2-docstring} src.policies.__all__
```

````

````{py:data} create_policy
:canonical: src.policies.create_policy
:value: >
   None

```{autodoc2-docstring} src.policies.create_policy
```

````
