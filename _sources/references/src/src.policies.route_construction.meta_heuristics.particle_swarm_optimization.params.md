# {py:mod}`src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params`

```{py:module} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PSOParams <src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams
    :summary:
    ```
````

### API

`````{py:class} PSOParams
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams
```

````{py:attribute} pop_size
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.pop_size
:type: int
:value: >
   30

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.pop_size
```

````

````{py:attribute} inertia_weight_start
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.inertia_weight_start
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.inertia_weight_start
```

````

````{py:attribute} inertia_weight_end
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.inertia_weight_end
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.inertia_weight_end
```

````

````{py:attribute} cognitive_coef
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.cognitive_coef
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.cognitive_coef
```

````

````{py:attribute} social_coef
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.social_coef
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.social_coef
```

````

````{py:attribute} position_min
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.position_min
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.position_min
```

````

````{py:attribute} position_max
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.position_max
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.position_max
```

````

````{py:attribute} velocity_max
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.velocity_max
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.velocity_max
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.profit_aware_operators
```

````

````{py:method} get_inertia_weight(iteration: int) -> float
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.get_inertia_weight

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.get_inertia_weight
```

````

````{py:property} c1
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.c1
:type: float

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.c1
```

````

````{py:property} c2
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.c2
:type: float

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.c2
```

````

````{py:property} w_start
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.w_start
:type: float

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.w_start
```

````

````{py:property} w_end
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.w_end
:type: float

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.w_end
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams.from_config
```

````

`````
