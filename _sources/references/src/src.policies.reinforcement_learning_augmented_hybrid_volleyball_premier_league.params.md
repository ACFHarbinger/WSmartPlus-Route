# {py:mod}`src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params`

```{py:module} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params
```

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLAHVPLParams <src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams>`
  - ```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams
    :summary:
    ```
````

### API

`````{py:class} RLAHVPLParams
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams
```

````{py:attribute} n_teams
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.n_teams
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.n_teams
```

````

````{py:attribute} time_limit
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.max_iterations
```

````

````{py:attribute} elite_coaching_max_iterations
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.elite_coaching_max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.elite_coaching_max_iterations
```

````

````{py:attribute} not_coached_max_iterations
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.not_coached_max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.not_coached_max_iterations
```

````

````{py:attribute} coaching_acceptance_threshold
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.coaching_acceptance_threshold
:type: float
:value: >
   1e-06

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.coaching_acceptance_threshold
```

````

````{py:attribute} sub_rate
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sub_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sub_rate
```

````

````{py:attribute} seed
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.seed
```

````

````{py:attribute} rl_config
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.rl_config
:type: logic.src.configs.policies.other.RLConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.rl_config
```

````

````{py:attribute} aco_params
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.aco_params
:type: src.policies.ant_colony_optimization_k_sparse.params.KSACOParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.aco_params
```

````

````{py:attribute} alns_params
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.alns_params
:type: src.policies.adaptive_large_neighborhood_search.params.ALNSParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.alns_params
```

````

````{py:attribute} hgs_params
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.hgs_params
:type: src.policies.hybrid_genetic_search.params.HGSParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.hgs_params
```

````

````{py:attribute} tabu_no_repeat_threshold
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.tabu_no_repeat_threshold
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.tabu_no_repeat_threshold
```

````

````{py:attribute} rts_params
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.rts_params
:type: src.policies.reactive_tabu_search.params.RTSParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.rts_params
```

````

````{py:attribute} gls_penalty_lambda
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.gls_penalty_lambda
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.gls_penalty_lambda
```

````

````{py:attribute} gls_penalty_alpha
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.gls_penalty_alpha
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.gls_penalty_alpha
```

````

````{py:attribute} gls_penalty_step
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.gls_penalty_step
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.gls_penalty_step
```

````

````{py:attribute} gls_probability
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.gls_probability
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.gls_probability
```

````

````{py:method} _get_val(category: str, key: str, default: typing.Any = None) -> typing.Any
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams._get_val

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams._get_val
```

````

````{py:method} _get_param(key: str, default: typing.Any = None) -> typing.Any
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams._get_param

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams._get_param
```

````

````{py:property} bandit_algorithm
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.bandit_algorithm
:type: str

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.bandit_algorithm
```

````

````{py:property} bandit_max_iterations
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.bandit_max_iterations
:type: int

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.bandit_max_iterations
```

````

````{py:property} bandit_quality_weight
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.bandit_quality_weight
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.bandit_quality_weight
```

````

````{py:property} bandit_improvement_weight
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.bandit_improvement_weight
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.bandit_improvement_weight
```

````

````{py:property} bandit_diversity_weight
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.bandit_diversity_weight
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.bandit_diversity_weight
```

````

````{py:property} bandit_novelty_weight
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.bandit_novelty_weight
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.bandit_novelty_weight
```

````

````{py:property} bandit_reward_threshold
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.bandit_reward_threshold
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.bandit_reward_threshold
```

````

````{py:property} bandit_default_reward
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.bandit_default_reward
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.bandit_default_reward
```

````

````{py:property} cfe_alpha
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_alpha
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_alpha
```

````

````{py:property} cfe_feature_dim
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_feature_dim
:type: int

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_feature_dim
```

````

````{py:property} cfe_operator_selection_threshold
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_operator_selection_threshold
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_operator_selection_threshold
```

````

````{py:property} cfe_lambda_prior
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_lambda_prior
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_lambda_prior
```

````

````{py:property} cfe_noise_variance
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_noise_variance
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_noise_variance
```

````

````{py:property} cfe_epsilon
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_epsilon
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_epsilon
```

````

````{py:property} cfe_epsilon_decay
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_epsilon_decay
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_epsilon_decay
```

````

````{py:property} cfe_epsilon_decay_step
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_epsilon_decay_step
:type: int

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_epsilon_decay_step
```

````

````{py:property} cfe_epsilon_min
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_epsilon_min
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_epsilon_min
```

````

````{py:property} cfe_diversity_history_size
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_diversity_history_size
:type: int

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_diversity_history_size
```

````

````{py:property} cfe_improvement_history_size
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_improvement_history_size
:type: int

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_improvement_history_size
```

````

````{py:property} cfe_operator_reward_size
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_operator_reward_size
:type: int

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_operator_reward_size
```

````

````{py:property} cfe_improvement_threshold
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_improvement_threshold
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.cfe_improvement_threshold
```

````

````{py:property} qlearning_alpha
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_alpha
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_alpha
```

````

````{py:property} qlearning_gamma
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_gamma
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_gamma
```

````

````{py:property} qlearning_epsilon
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_epsilon
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_epsilon
```

````

````{py:property} qlearning_epsilon_decay
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_epsilon_decay
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_epsilon_decay
```

````

````{py:property} qlearning_epsilon_decay_step
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_epsilon_decay_step
:type: int

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_epsilon_decay_step
```

````

````{py:property} qlearning_epsilon_min
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_epsilon_min
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_epsilon_min
```

````

````{py:property} qlearning_history_size
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_history_size
:type: int

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_history_size
```

````

````{py:property} qlearning_rewards_size
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_rewards_size
:type: int

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_rewards_size
```

````

````{py:property} qlearning_improvement_thresholds
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_improvement_thresholds
:type: typing.Tuple[float, float]

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.qlearning_improvement_thresholds
```

````

````{py:property} sarsa_alpha
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_alpha
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_alpha
```

````

````{py:property} sarsa_gamma
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_gamma
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_gamma
```

````

````{py:property} sarsa_epsilon
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_epsilon
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_epsilon
```

````

````{py:property} sarsa_epsilon_decay
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_epsilon_decay
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_epsilon_decay
```

````

````{py:property} sarsa_epsilon_decay_step
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_epsilon_decay_step
:type: int

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_epsilon_decay_step
```

````

````{py:property} sarsa_epsilon_min
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_epsilon_min
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_epsilon_min
```

````

````{py:property} sarsa_diversity_size
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_diversity_size
:type: int

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_diversity_size
```

````

````{py:property} sarsa_scores_size
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_scores_size
:type: int

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_scores_size
```

````

````{py:property} sarsa_qtable_size_rate
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_qtable_size_rate
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_qtable_size_rate
```

````

````{py:property} sarsa_improvement_thresholds
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_improvement_thresholds
:type: typing.Tuple[float, float]

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_improvement_thresholds
```

````

````{py:property} sarsa_operator_progress_thresholds
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_operator_progress_thresholds
:type: typing.Tuple[float, float]

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_operator_progress_thresholds
```

````

````{py:property} sarsa_operator_stagnation_thresholds
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_operator_stagnation_thresholds
:type: typing.Tuple[int, int]

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_operator_stagnation_thresholds
```

````

````{py:property} sarsa_operator_diversity_thresholds
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_operator_diversity_thresholds
:type: typing.Tuple[float, float]

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams.sarsa_operator_diversity_thresholds
```

````

`````
