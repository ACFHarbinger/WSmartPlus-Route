# {py:mod}`src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params`

```{py:module} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSALNSParams <src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams
    :summary:
    ```
````

### API

`````{py:class} HGSALNSParams
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams
```

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams.profit_aware_operators
```

````

````{py:attribute} hgs_params
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams.hgs_params
:type: logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.params.HGSParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams.hgs_params
```

````

````{py:attribute} alns_params
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams.alns_params
:type: logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params.ALNSParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams.alns_params
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params.HGSALNSParams.to_dict
```

````

`````
