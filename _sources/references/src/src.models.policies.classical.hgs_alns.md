# {py:mod}`src.models.policies.classical.hgs_alns`

```{py:module} src.models.policies.classical.hgs_alns
```

```{autodoc2-docstring} src.models.policies.classical.hgs_alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSALNSPolicy <src.models.policies.classical.hgs_alns.HGSALNSPolicy>`
  - ```{autodoc2-docstring} src.models.policies.classical.hgs_alns.HGSALNSPolicy
    :summary:
    ```
````

### API

`````{py:class} HGSALNSPolicy(**data: typing.Any)
:canonical: src.models.policies.classical.hgs_alns.HGSALNSPolicy

Bases: {py:obj}`logic.src.models.policies.classical.hgs.VectorizedHGS`, {py:obj}`pydantic.BaseModel`

```{autodoc2-docstring} src.models.policies.classical.hgs_alns.HGSALNSPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.classical.hgs_alns.HGSALNSPolicy.__init__
```

````{py:attribute} model_config
:canonical: src.models.policies.classical.hgs_alns.HGSALNSPolicy.model_config
:value: >
   'ConfigDict(...)'

```{autodoc2-docstring} src.models.policies.classical.hgs_alns.HGSALNSPolicy.model_config
```

````

````{py:attribute} env_name
:canonical: src.models.policies.classical.hgs_alns.HGSALNSPolicy.env_name
:type: str
:value: >
   'Field(...)'

```{autodoc2-docstring} src.models.policies.classical.hgs_alns.HGSALNSPolicy.env_name
```

````

````{py:attribute} time_limit
:canonical: src.models.policies.classical.hgs_alns.HGSALNSPolicy.time_limit
:type: float
:value: >
   'Field(...)'

```{autodoc2-docstring} src.models.policies.classical.hgs_alns.HGSALNSPolicy.time_limit
```

````

````{py:attribute} population_size
:canonical: src.models.policies.classical.hgs_alns.HGSALNSPolicy.population_size
:type: int
:value: >
   'Field(...)'

```{autodoc2-docstring} src.models.policies.classical.hgs_alns.HGSALNSPolicy.population_size
```

````

````{py:attribute} n_generations
:canonical: src.models.policies.classical.hgs_alns.HGSALNSPolicy.n_generations
:type: int
:value: >
   'Field(...)'

```{autodoc2-docstring} src.models.policies.classical.hgs_alns.HGSALNSPolicy.n_generations
```

````

````{py:attribute} elite_size
:canonical: src.models.policies.classical.hgs_alns.HGSALNSPolicy.elite_size
:type: int
:value: >
   'Field(...)'

```{autodoc2-docstring} src.models.policies.classical.hgs_alns.HGSALNSPolicy.elite_size
```

````

````{py:attribute} max_vehicles
:canonical: src.models.policies.classical.hgs_alns.HGSALNSPolicy.max_vehicles
:type: int
:value: >
   'Field(...)'

```{autodoc2-docstring} src.models.policies.classical.hgs_alns.HGSALNSPolicy.max_vehicles
```

````

````{py:attribute} alns_education_iterations
:canonical: src.models.policies.classical.hgs_alns.HGSALNSPolicy.alns_education_iterations
:type: int
:value: >
   'Field(...)'

```{autodoc2-docstring} src.models.policies.classical.hgs_alns.HGSALNSPolicy.alns_education_iterations
```

````

````{py:method} solve(dist_matrix, demands, capacity, **kwargs)
:canonical: src.models.policies.classical.hgs_alns.HGSALNSPolicy.solve

```{autodoc2-docstring} src.models.policies.classical.hgs_alns.HGSALNSPolicy.solve
```

````

`````
