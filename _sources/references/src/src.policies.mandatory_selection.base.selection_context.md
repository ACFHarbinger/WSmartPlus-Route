# {py:mod}`src.policies.mandatory_selection.base.selection_context`

```{py:module} src.policies.mandatory_selection.base.selection_context
```

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SelectionContext <src.policies.mandatory_selection.base.selection_context.SelectionContext>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext
    :summary:
    ```
````

### API

`````{py:class} SelectionContext
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext
```

````{py:attribute} bin_ids
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.bin_ids
:type: numpy.typing.NDArray[numpy.int32]
:value: >
   None

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.bin_ids
```

````

````{py:attribute} current_fill
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.current_fill
:type: numpy.typing.NDArray[numpy.float64]
:value: >
   None

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.current_fill
```

````

````{py:attribute} accumulation_rates
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.accumulation_rates
:type: typing.Optional[numpy.typing.NDArray[numpy.float64]]
:value: >
   None

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.accumulation_rates
```

````

````{py:attribute} std_deviations
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.std_deviations
:type: typing.Optional[numpy.typing.NDArray[numpy.float64]]
:value: >
   None

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.std_deviations
```

````

````{py:attribute} current_day
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.current_day
:type: int
:value: >
   0

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.current_day
```

````

````{py:attribute} threshold
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.threshold
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.threshold
```

````

````{py:attribute} next_collection_day
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.next_collection_day
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.next_collection_day
```

````

````{py:attribute} distance_matrix
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.distance_matrix
:type: typing.Optional[numpy.typing.NDArray[typing.Any]]
:value: >
   None

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.distance_matrix
```

````

````{py:attribute} paths_between_states
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.paths_between_states
:type: typing.Optional[typing.List[typing.List[typing.List[int]]]]
:value: >
   None

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.paths_between_states
```

````

````{py:attribute} vehicle_capacity
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.vehicle_capacity
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.vehicle_capacity
```

````

````{py:attribute} revenue_kg
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.revenue_kg
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.revenue_kg
```

````

````{py:attribute} bin_density
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.bin_density
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.bin_density
```

````

````{py:attribute} bin_volume
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.bin_volume
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.bin_volume
```

````

````{py:attribute} max_fill
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.max_fill
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.max_fill
```

````

````{py:attribute} horizon_days
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.horizon_days
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.horizon_days
```

````

````{py:attribute} critical_threshold
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.critical_threshold
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.critical_threshold
```

````

````{py:attribute} synergy_threshold
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.synergy_threshold
:type: float
:value: >
   0.6

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.synergy_threshold
```

````

````{py:attribute} radius
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.radius
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.radius
```

````

````{py:attribute} n_vehicles
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.n_vehicles
:type: int
:value: >
   1

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.n_vehicles
```

````

````{py:attribute} cost_per_km
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.cost_per_km
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.cost_per_km
```

````

````{py:attribute} use_eoq_threshold
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.use_eoq_threshold
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.use_eoq_threshold
```

````

````{py:attribute} holding_cost_per_kg_day
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.holding_cost_per_kg_day
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.holding_cost_per_kg_day
```

````

````{py:attribute} ordering_cost_per_visit
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.ordering_cost_per_visit
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.ordering_cost_per_visit
```

````

````{py:attribute} rollout_horizon
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.rollout_horizon
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.rollout_horizon
```

````

````{py:attribute} rollout_base_policy
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.rollout_base_policy
:type: str
:value: >
   'last_minute'

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.rollout_base_policy
```

````

````{py:attribute} rollout_n_scenarios
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.rollout_n_scenarios
:type: int
:value: >
   1

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.rollout_n_scenarios
```

````

````{py:attribute} rollout_discount
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.rollout_discount
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.rollout_discount
```

````

````{py:attribute} whittle_discount
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.whittle_discount
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.whittle_discount
```

````

````{py:attribute} whittle_grid_size
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.whittle_grid_size
:type: int
:value: >
   21

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.whittle_grid_size
```

````

````{py:attribute} cvar_alpha
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.cvar_alpha
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.cvar_alpha
```

````

````{py:attribute} savings_min_fill_ratio
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.savings_min_fill_ratio
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.savings_min_fill_ratio
```

````

````{py:attribute} service_radius
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.service_radius
:type: float
:value: >
   5.0

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.service_radius
```

````

````{py:attribute} modular_alpha
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.modular_alpha
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.modular_alpha
```

````

````{py:attribute} modular_budget
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.modular_budget
:type: int
:value: >
   0

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.modular_budget
```

````

````{py:attribute} learned_model_path
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.learned_model_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.learned_model_path
```

````

````{py:attribute} learned_threshold
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.learned_threshold
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.learned_threshold
```

````

````{py:attribute} dispatcher_state_path
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.dispatcher_state_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.dispatcher_state_path
```

````

````{py:attribute} dispatcher_candidate_strategies
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.dispatcher_candidate_strategies
:type: typing.Optional[typing.List[str]]
:value: >
   None

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.dispatcher_candidate_strategies
```

````

````{py:attribute} dispatcher_exploration
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.dispatcher_exploration
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.dispatcher_exploration
```

````

````{py:attribute} dispatcher_mode
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.dispatcher_mode
:type: str
:value: >
   'union'

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.dispatcher_mode
```

````

````{py:attribute} wasserstein_radius
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.wasserstein_radius
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.wasserstein_radius
```

````

````{py:attribute} wasserstein_p
:canonical: src.policies.mandatory_selection.base.selection_context.SelectionContext.wasserstein_p
:type: int
:value: >
   1

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_context.SelectionContext.wasserstein_p
```

````

`````
