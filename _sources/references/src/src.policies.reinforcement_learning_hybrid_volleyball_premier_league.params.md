# {py:mod}`src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params`

```{py:module} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params
```

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLHVPLParams <src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams>`
  - ```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams
    :summary:
    ```
````

### API

`````{py:class} RLHVPLParams
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams
```

````{py:attribute} n_teams
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.n_teams
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.n_teams
```

````

````{py:attribute} max_iterations
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.max_iterations
```

````

````{py:attribute} sub_rate
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sub_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sub_rate
```

````

````{py:attribute} time_limit
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.seed
```

````

````{py:attribute} rl_config
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.rl_config
:type: logic.src.configs.policies.other.RLConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.rl_config
```

````

````{py:attribute} aco_params
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.aco_params
:type: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.aco_params
```

````

````{py:attribute} alns_params
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.alns_params
:type: src.policies.adaptive_large_neighborhood_search.params.ALNSParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.alns_params
```

````

````{py:attribute} pheromone_update_strategy
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.pheromone_update_strategy
:type: str
:value: >
   'profit'

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.pheromone_update_strategy
```

````

````{py:attribute} profit_weight
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.profit_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.profit_weight
```

````

````{py:attribute} elite_coaching_iterations
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.elite_coaching_iterations
:type: int
:value: >
   300

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.elite_coaching_iterations
```

````

````{py:attribute} regular_coaching_iterations
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.regular_coaching_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.regular_coaching_iterations
```

````

````{py:attribute} elite_size
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.elite_size
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.elite_size
```

````

````{py:property} qlearning_alpha
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.qlearning_alpha
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.qlearning_alpha
```

````

````{py:property} qlearning_gamma
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.qlearning_gamma
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.qlearning_gamma
```

````

````{py:property} qlearning_epsilon
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.qlearning_epsilon
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.qlearning_epsilon
```

````

````{py:property} qlearning_epsilon_decay
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.qlearning_epsilon_decay
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.qlearning_epsilon_decay
```

````

````{py:property} qlearning_epsilon_decay_step
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.qlearning_epsilon_decay_step
:type: int

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.qlearning_epsilon_decay_step
```

````

````{py:property} qlearning_epsilon_min
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.qlearning_epsilon_min
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.qlearning_epsilon_min
```

````

````{py:property} qlearning_improvement_thresholds
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.qlearning_improvement_thresholds
:type: typing.Tuple[float, float]

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.qlearning_improvement_thresholds
```

````

````{py:property} sarsa_alpha
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_alpha
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_alpha
```

````

````{py:property} sarsa_gamma
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_gamma
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_gamma
```

````

````{py:property} sarsa_epsilon
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_epsilon
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_epsilon
```

````

````{py:property} sarsa_epsilon_decay
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_epsilon_decay
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_epsilon_decay
```

````

````{py:property} sarsa_epsilon_decay_step
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_epsilon_decay_step
:type: int

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_epsilon_decay_step
```

````

````{py:property} sarsa_epsilon_min
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_epsilon_min
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_epsilon_min
```

````

````{py:property} sarsa_diversity_size
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_diversity_size
:type: int

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_diversity_size
```

````

````{py:property} sarsa_improvement_thresholds
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_improvement_thresholds
:type: typing.Tuple[float, float]

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_improvement_thresholds
```

````

````{py:property} sarsa_operator_progress_thresholds
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_operator_progress_thresholds
:type: typing.Tuple[float, float]

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_operator_progress_thresholds
```

````

````{py:property} sarsa_operator_stagnation_thresholds
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_operator_stagnation_thresholds
:type: typing.Tuple[int, int]

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_operator_stagnation_thresholds
```

````

````{py:property} sarsa_operator_diversity_thresholds
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_operator_diversity_thresholds
:type: typing.Tuple[float, float]

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams.sarsa_operator_diversity_thresholds
```

````

`````
