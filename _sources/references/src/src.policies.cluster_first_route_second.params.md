# {py:mod}`src.policies.cluster_first_route_second.params`

```{py:module} src.policies.cluster_first_route_second.params
```

```{autodoc2-docstring} src.policies.cluster_first_route_second.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CFRSParams <src.policies.cluster_first_route_second.params.CFRSParams>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.params.CFRSParams
    :summary:
    ```
````

### API

`````{py:class} CFRSParams
:canonical: src.policies.cluster_first_route_second.params.CFRSParams

```{autodoc2-docstring} src.policies.cluster_first_route_second.params.CFRSParams
```

````{py:attribute} num_clusters
:canonical: src.policies.cluster_first_route_second.params.CFRSParams.num_clusters
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.cluster_first_route_second.params.CFRSParams.num_clusters
```

````

````{py:attribute} assignment_method
:canonical: src.policies.cluster_first_route_second.params.CFRSParams.assignment_method
:type: str
:value: >
   'angular'

```{autodoc2-docstring} src.policies.cluster_first_route_second.params.CFRSParams.assignment_method
```

````

````{py:attribute} route_optimizer
:canonical: src.policies.cluster_first_route_second.params.CFRSParams.route_optimizer
:type: str
:value: >
   'lkh'

```{autodoc2-docstring} src.policies.cluster_first_route_second.params.CFRSParams.route_optimizer
```

````

````{py:attribute} strict_fleet
:canonical: src.policies.cluster_first_route_second.params.CFRSParams.strict_fleet
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.cluster_first_route_second.params.CFRSParams.strict_fleet
```

````

````{py:attribute} seed_criterion
:canonical: src.policies.cluster_first_route_second.params.CFRSParams.seed_criterion
:type: str
:value: >
   'max_dist'

```{autodoc2-docstring} src.policies.cluster_first_route_second.params.CFRSParams.seed_criterion
```

````

````{py:attribute} mip_objective
:canonical: src.policies.cluster_first_route_second.params.CFRSParams.mip_objective
:type: str
:value: >
   'distance'

```{autodoc2-docstring} src.policies.cluster_first_route_second.params.CFRSParams.mip_objective
```

````

````{py:attribute} time_limit
:canonical: src.policies.cluster_first_route_second.params.CFRSParams.time_limit
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.cluster_first_route_second.params.CFRSParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.cluster_first_route_second.params.CFRSParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.cluster_first_route_second.params.CFRSParams.seed
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.cluster_first_route_second.params.CFRSParams
:canonical: src.policies.cluster_first_route_second.params.CFRSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.cluster_first_route_second.params.CFRSParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.cluster_first_route_second.params.CFRSParams.to_dict

```{autodoc2-docstring} src.policies.cluster_first_route_second.params.CFRSParams.to_dict
```

````

`````
