# {py:mod}`src.configs.policies.other.must_go`

```{py:module} src.configs.policies.other.must_go
```

```{autodoc2-docstring} src.configs.policies.other.must_go
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MustGoConfig <src.configs.policies.other.must_go.MustGoConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig
    :summary:
    ```
````

### API

`````{py:class} MustGoConfig
:canonical: src.configs.policies.other.must_go.MustGoConfig

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig
```

````{py:attribute} strategy
:canonical: src.configs.policies.other.must_go.MustGoConfig.strategy
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.strategy
```

````

````{py:attribute} threshold
:canonical: src.configs.policies.other.must_go.MustGoConfig.threshold
:type: float
:value: >
   0.7

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.threshold
```

````

````{py:attribute} frequency
:canonical: src.configs.policies.other.must_go.MustGoConfig.frequency
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.frequency
```

````

````{py:attribute} confidence_factor
:canonical: src.configs.policies.other.must_go.MustGoConfig.confidence_factor
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.confidence_factor
```

````

````{py:attribute} revenue_kg
:canonical: src.configs.policies.other.must_go.MustGoConfig.revenue_kg
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.revenue_kg
```

````

````{py:attribute} bin_capacity
:canonical: src.configs.policies.other.must_go.MustGoConfig.bin_capacity
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.bin_capacity
```

````

````{py:attribute} revenue_threshold
:canonical: src.configs.policies.other.must_go.MustGoConfig.revenue_threshold
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.revenue_threshold
```

````

````{py:attribute} max_fill
:canonical: src.configs.policies.other.must_go.MustGoConfig.max_fill
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.max_fill
```

````

````{py:attribute} horizon_days
:canonical: src.configs.policies.other.must_go.MustGoConfig.horizon_days
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.horizon_days
```

````

````{py:attribute} critical_threshold
:canonical: src.configs.policies.other.must_go.MustGoConfig.critical_threshold
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.critical_threshold
```

````

````{py:attribute} synergy_threshold
:canonical: src.configs.policies.other.must_go.MustGoConfig.synergy_threshold
:type: float
:value: >
   0.6

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.synergy_threshold
```

````

````{py:attribute} radius
:canonical: src.configs.policies.other.must_go.MustGoConfig.radius
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.radius
```

````

````{py:attribute} combined_strategies
:canonical: src.configs.policies.other.must_go.MustGoConfig.combined_strategies
:type: typing.Optional[list]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.combined_strategies
```

````

````{py:attribute} logic
:canonical: src.configs.policies.other.must_go.MustGoConfig.logic
:type: str
:value: >
   'or'

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.logic
```

````

````{py:attribute} hidden_dim
:canonical: src.configs.policies.other.must_go.MustGoConfig.hidden_dim
:type: int
:value: >
   128

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.hidden_dim
```

````

````{py:attribute} lstm_hidden
:canonical: src.configs.policies.other.must_go.MustGoConfig.lstm_hidden
:type: int
:value: >
   64

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.lstm_hidden
```

````

````{py:attribute} history_length
:canonical: src.configs.policies.other.must_go.MustGoConfig.history_length
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.history_length
```

````

````{py:attribute} manager_critical_threshold
:canonical: src.configs.policies.other.must_go.MustGoConfig.manager_critical_threshold
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.manager_critical_threshold
```

````

````{py:attribute} manager_weights
:canonical: src.configs.policies.other.must_go.MustGoConfig.manager_weights
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.manager_weights
```

````

````{py:attribute} device
:canonical: src.configs.policies.other.must_go.MustGoConfig.device
:type: str
:value: >
   'cuda'

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.device
```

````

````{py:attribute} params
:canonical: src.configs.policies.other.must_go.MustGoConfig.params
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.params
```

````

````{py:attribute} n_vehicles
:canonical: src.configs.policies.other.must_go.MustGoConfig.n_vehicles
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.n_vehicles
```

````

````{py:attribute} cost_per_km
:canonical: src.configs.policies.other.must_go.MustGoConfig.cost_per_km
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.cost_per_km
```

````

````{py:attribute} use_eoq_threshold
:canonical: src.configs.policies.other.must_go.MustGoConfig.use_eoq_threshold
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.use_eoq_threshold
```

````

````{py:attribute} holding_cost_per_kg_day
:canonical: src.configs.policies.other.must_go.MustGoConfig.holding_cost_per_kg_day
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.holding_cost_per_kg_day
```

````

````{py:attribute} ordering_cost_per_visit
:canonical: src.configs.policies.other.must_go.MustGoConfig.ordering_cost_per_visit
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.ordering_cost_per_visit
```

````

````{py:attribute} rollout_horizon
:canonical: src.configs.policies.other.must_go.MustGoConfig.rollout_horizon
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.rollout_horizon
```

````

````{py:attribute} rollout_base_policy
:canonical: src.configs.policies.other.must_go.MustGoConfig.rollout_base_policy
:type: str
:value: >
   'last_minute'

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.rollout_base_policy
```

````

````{py:attribute} rollout_n_scenarios
:canonical: src.configs.policies.other.must_go.MustGoConfig.rollout_n_scenarios
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.rollout_n_scenarios
```

````

````{py:attribute} rollout_discount
:canonical: src.configs.policies.other.must_go.MustGoConfig.rollout_discount
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.rollout_discount
```

````

````{py:attribute} whittle_discount
:canonical: src.configs.policies.other.must_go.MustGoConfig.whittle_discount
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.whittle_discount
```

````

````{py:attribute} whittle_grid_size
:canonical: src.configs.policies.other.must_go.MustGoConfig.whittle_grid_size
:type: int
:value: >
   21

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.whittle_grid_size
```

````

````{py:attribute} cvar_alpha
:canonical: src.configs.policies.other.must_go.MustGoConfig.cvar_alpha
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.cvar_alpha
```

````

````{py:attribute} savings_min_fill_ratio
:canonical: src.configs.policies.other.must_go.MustGoConfig.savings_min_fill_ratio
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.savings_min_fill_ratio
```

````

````{py:attribute} service_radius
:canonical: src.configs.policies.other.must_go.MustGoConfig.service_radius
:type: float
:value: >
   5.0

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.service_radius
```

````

````{py:attribute} modular_alpha
:canonical: src.configs.policies.other.must_go.MustGoConfig.modular_alpha
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.modular_alpha
```

````

````{py:attribute} modular_budget
:canonical: src.configs.policies.other.must_go.MustGoConfig.modular_budget
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.modular_budget
```

````

````{py:attribute} learned_model_path
:canonical: src.configs.policies.other.must_go.MustGoConfig.learned_model_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.learned_model_path
```

````

````{py:attribute} learned_threshold
:canonical: src.configs.policies.other.must_go.MustGoConfig.learned_threshold
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.learned_threshold
```

````

````{py:attribute} dispatcher_state_path
:canonical: src.configs.policies.other.must_go.MustGoConfig.dispatcher_state_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.dispatcher_state_path
```

````

````{py:attribute} dispatcher_candidate_strategies
:canonical: src.configs.policies.other.must_go.MustGoConfig.dispatcher_candidate_strategies
:type: typing.Optional[typing.List[str]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.dispatcher_candidate_strategies
```

````

````{py:attribute} dispatcher_exploration
:canonical: src.configs.policies.other.must_go.MustGoConfig.dispatcher_exploration
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.dispatcher_exploration
```

````

````{py:attribute} dispatcher_mode
:canonical: src.configs.policies.other.must_go.MustGoConfig.dispatcher_mode
:type: str
:value: >
   'union'

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.dispatcher_mode
```

````

````{py:attribute} wasserstein_radius
:canonical: src.configs.policies.other.must_go.MustGoConfig.wasserstein_radius
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.wasserstein_radius
```

````

````{py:attribute} wasserstein_p
:canonical: src.configs.policies.other.must_go.MustGoConfig.wasserstein_p
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.policies.other.must_go.MustGoConfig.wasserstein_p
```

````

`````
