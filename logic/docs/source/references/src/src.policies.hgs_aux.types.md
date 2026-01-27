# {py:mod}`src.policies.hgs_aux.types`

```{py:module} src.policies.hgs_aux.types
```

```{autodoc2-docstring} src.policies.hgs_aux.types
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Individual <src.policies.hgs_aux.types.Individual>`
  - ```{autodoc2-docstring} src.policies.hgs_aux.types.Individual
    :summary:
    ```
* - {py:obj}`HGSParams <src.policies.hgs_aux.types.HGSParams>`
  - ```{autodoc2-docstring} src.policies.hgs_aux.types.HGSParams
    :summary:
    ```
````

### API

`````{py:class} Individual(giant_tour: list[int])
:canonical: src.policies.hgs_aux.types.Individual

```{autodoc2-docstring} src.policies.hgs_aux.types.Individual
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hgs_aux.types.Individual.__init__
```

````{py:method} __lt__(other: src.policies.hgs_aux.types.Individual) -> bool
:canonical: src.policies.hgs_aux.types.Individual.__lt__

```{autodoc2-docstring} src.policies.hgs_aux.types.Individual.__lt__
```

````

`````

````{py:class} HGSParams(time_limit: int = 10, population_size: int = 50, elite_size: int = 10, mutation_rate: float = 0.2, max_vehicles: int = 0)
:canonical: src.policies.hgs_aux.types.HGSParams

```{autodoc2-docstring} src.policies.hgs_aux.types.HGSParams
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hgs_aux.types.HGSParams.__init__
```

````
