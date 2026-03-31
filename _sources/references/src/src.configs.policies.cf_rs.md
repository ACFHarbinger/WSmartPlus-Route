# {py:mod}`src.configs.policies.cf_rs`

```{py:module} src.configs.policies.cf_rs
```

```{autodoc2-docstring} src.configs.policies.cf_rs
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CFRSConfig <src.configs.policies.cf_rs.CFRSConfig>`
  - ```{autodoc2-docstring} src.configs.policies.cf_rs.CFRSConfig
    :summary:
    ```
````

### API

`````{py:class} CFRSConfig
:canonical: src.configs.policies.cf_rs.CFRSConfig

```{autodoc2-docstring} src.configs.policies.cf_rs.CFRSConfig
```

````{py:attribute} vrpp
:canonical: src.configs.policies.cf_rs.CFRSConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.cf_rs.CFRSConfig.vrpp
```

````

````{py:attribute} seed
:canonical: src.configs.policies.cf_rs.CFRSConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.cf_rs.CFRSConfig.seed
```

````

````{py:attribute} num_clusters
:canonical: src.configs.policies.cf_rs.CFRSConfig.num_clusters
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.policies.cf_rs.CFRSConfig.num_clusters
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.cf_rs.CFRSConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.cf_rs.CFRSConfig.time_limit
```

````

````{py:attribute} assignment_method
:canonical: src.configs.policies.cf_rs.CFRSConfig.assignment_method
:type: str
:value: >
   'greedy'

```{autodoc2-docstring} src.configs.policies.cf_rs.CFRSConfig.assignment_method
```

````

````{py:attribute} route_optimizer
:canonical: src.configs.policies.cf_rs.CFRSConfig.route_optimizer
:type: str
:value: >
   'default'

```{autodoc2-docstring} src.configs.policies.cf_rs.CFRSConfig.route_optimizer
```

````

````{py:attribute} strict_fleet
:canonical: src.configs.policies.cf_rs.CFRSConfig.strict_fleet
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.cf_rs.CFRSConfig.strict_fleet
```

````

````{py:attribute} seed_criterion
:canonical: src.configs.policies.cf_rs.CFRSConfig.seed_criterion
:type: str
:value: >
   'distance'

```{autodoc2-docstring} src.configs.policies.cf_rs.CFRSConfig.seed_criterion
```

````

````{py:attribute} mip_objective
:canonical: src.configs.policies.cf_rs.CFRSConfig.mip_objective
:type: str
:value: >
   'minimize_cost'

```{autodoc2-docstring} src.configs.policies.cf_rs.CFRSConfig.mip_objective
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.cf_rs.CFRSConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.cf_rs.CFRSConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.cf_rs.CFRSConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.cf_rs.CFRSConfig.post_processing
```

````

`````
