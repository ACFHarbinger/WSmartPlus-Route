# {py:mod}`src.configs.policies.other.reinforcement_learning`

```{py:module} src.configs.policies.other.reinforcement_learning
```

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BanditConfig <src.configs.policies.other.reinforcement_learning.BanditConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.BanditConfig
    :summary:
    ```
* - {py:obj}`TDLearningConfig <src.configs.policies.other.reinforcement_learning.TDLearningConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.TDLearningConfig
    :summary:
    ```
* - {py:obj}`LinUCBConfig <src.configs.policies.other.reinforcement_learning.LinUCBConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.LinUCBConfig
    :summary:
    ```
* - {py:obj}`EvolutionaryCMABConfig <src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig
    :summary:
    ```
* - {py:obj}`RewardShapingConfig <src.configs.policies.other.reinforcement_learning.RewardShapingConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RewardShapingConfig
    :summary:
    ```
* - {py:obj}`FeatureExtractorConfig <src.configs.policies.other.reinforcement_learning.FeatureExtractorConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.FeatureExtractorConfig
    :summary:
    ```
* - {py:obj}`ContextFeatureExtractorConfig <src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig
    :summary:
    ```
* - {py:obj}`GPCMABConfig <src.configs.policies.other.reinforcement_learning.GPCMABConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.GPCMABConfig
    :summary:
    ```
* - {py:obj}`RLConfig <src.configs.policies.other.reinforcement_learning.RLConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RLConfig
    :summary:
    ```
````

### API

`````{py:class} BanditConfig
:canonical: src.configs.policies.other.reinforcement_learning.BanditConfig

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.BanditConfig
```

````{py:attribute} algorithm
:canonical: src.configs.policies.other.reinforcement_learning.BanditConfig.algorithm
:type: str
:value: >
   'ucb1'

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.BanditConfig.algorithm
```

````

````{py:attribute} epsilon
:canonical: src.configs.policies.other.reinforcement_learning.BanditConfig.epsilon
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.BanditConfig.epsilon
```

````

````{py:attribute} epsilon_decay
:canonical: src.configs.policies.other.reinforcement_learning.BanditConfig.epsilon_decay
:type: float
:value: >
   0.999

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.BanditConfig.epsilon_decay
```

````

````{py:attribute} epsilon_min
:canonical: src.configs.policies.other.reinforcement_learning.BanditConfig.epsilon_min
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.BanditConfig.epsilon_min
```

````

````{py:attribute} c
:canonical: src.configs.policies.other.reinforcement_learning.BanditConfig.c
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.BanditConfig.c
```

````

````{py:attribute} temperature
:canonical: src.configs.policies.other.reinforcement_learning.BanditConfig.temperature
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.BanditConfig.temperature
```

````

````{py:attribute} alpha_prior
:canonical: src.configs.policies.other.reinforcement_learning.BanditConfig.alpha_prior
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.BanditConfig.alpha_prior
```

````

````{py:attribute} beta_prior
:canonical: src.configs.policies.other.reinforcement_learning.BanditConfig.beta_prior
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.BanditConfig.beta_prior
```

````

````{py:attribute} gamma
:canonical: src.configs.policies.other.reinforcement_learning.BanditConfig.gamma
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.BanditConfig.gamma
```

````

````{py:attribute} window_size
:canonical: src.configs.policies.other.reinforcement_learning.BanditConfig.window_size
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.BanditConfig.window_size
```

````

````{py:attribute} history_size
:canonical: src.configs.policies.other.reinforcement_learning.BanditConfig.history_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.BanditConfig.history_size
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.reinforcement_learning.BanditConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.BanditConfig.seed
```

````

`````

`````{py:class} TDLearningConfig
:canonical: src.configs.policies.other.reinforcement_learning.TDLearningConfig

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.TDLearningConfig
```

````{py:attribute} algorithm
:canonical: src.configs.policies.other.reinforcement_learning.TDLearningConfig.algorithm
:type: str
:value: >
   'q_learning'

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.TDLearningConfig.algorithm
```

````

````{py:attribute} alpha
:canonical: src.configs.policies.other.reinforcement_learning.TDLearningConfig.alpha
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.TDLearningConfig.alpha
```

````

````{py:attribute} gamma
:canonical: src.configs.policies.other.reinforcement_learning.TDLearningConfig.gamma
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.TDLearningConfig.gamma
```

````

````{py:attribute} epsilon
:canonical: src.configs.policies.other.reinforcement_learning.TDLearningConfig.epsilon
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.TDLearningConfig.epsilon
```

````

````{py:attribute} epsilon_decay
:canonical: src.configs.policies.other.reinforcement_learning.TDLearningConfig.epsilon_decay
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.TDLearningConfig.epsilon_decay
```

````

````{py:attribute} epsilon_min
:canonical: src.configs.policies.other.reinforcement_learning.TDLearningConfig.epsilon_min
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.TDLearningConfig.epsilon_min
```

````

````{py:attribute} history_size
:canonical: src.configs.policies.other.reinforcement_learning.TDLearningConfig.history_size
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.TDLearningConfig.history_size
```

````

````{py:attribute} n_states
:canonical: src.configs.policies.other.reinforcement_learning.TDLearningConfig.n_states
:type: int
:value: >
   27

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.TDLearningConfig.n_states
```

````

````{py:attribute} n_actions
:canonical: src.configs.policies.other.reinforcement_learning.TDLearningConfig.n_actions
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.TDLearningConfig.n_actions
```

````

`````

`````{py:class} LinUCBConfig
:canonical: src.configs.policies.other.reinforcement_learning.LinUCBConfig

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.LinUCBConfig
```

````{py:attribute} alpha
:canonical: src.configs.policies.other.reinforcement_learning.LinUCBConfig.alpha
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.LinUCBConfig.alpha
```

````

````{py:attribute} feature_dim
:canonical: src.configs.policies.other.reinforcement_learning.LinUCBConfig.feature_dim
:type: int
:value: >
   8

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.LinUCBConfig.feature_dim
```

````

````{py:attribute} lambda_prior
:canonical: src.configs.policies.other.reinforcement_learning.LinUCBConfig.lambda_prior
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.LinUCBConfig.lambda_prior
```

````

````{py:attribute} noise_variance
:canonical: src.configs.policies.other.reinforcement_learning.LinUCBConfig.noise_variance
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.LinUCBConfig.noise_variance
```

````

````{py:attribute} history_size
:canonical: src.configs.policies.other.reinforcement_learning.LinUCBConfig.history_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.LinUCBConfig.history_size
```

````

`````

`````{py:class} EvolutionaryCMABConfig
:canonical: src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig
```

````{py:attribute} quality_weight
:canonical: src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig.quality_weight
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig.quality_weight
```

````

````{py:attribute} improvement_weight
:canonical: src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig.improvement_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig.improvement_weight
```

````

````{py:attribute} diversity_weight
:canonical: src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig.diversity_weight
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig.diversity_weight
```

````

````{py:attribute} novelty_weight
:canonical: src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig.novelty_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig.novelty_weight
```

````

````{py:attribute} reward_threshold
:canonical: src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig.reward_threshold
:type: float
:value: >
   1e-06

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig.reward_threshold
```

````

````{py:attribute} default_reward
:canonical: src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig.default_reward
:type: float
:value: >
   5.0

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig.default_reward
```

````

`````

`````{py:class} RewardShapingConfig
:canonical: src.configs.policies.other.reinforcement_learning.RewardShapingConfig

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RewardShapingConfig
```

````{py:attribute} best_reward
:canonical: src.configs.policies.other.reinforcement_learning.RewardShapingConfig.best_reward
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RewardShapingConfig.best_reward
```

````

````{py:attribute} local_reward
:canonical: src.configs.policies.other.reinforcement_learning.RewardShapingConfig.local_reward
:type: float
:value: >
   5.0

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RewardShapingConfig.local_reward
```

````

````{py:attribute} accepted_reward
:canonical: src.configs.policies.other.reinforcement_learning.RewardShapingConfig.accepted_reward
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RewardShapingConfig.accepted_reward
```

````

````{py:attribute} rejected_reward
:canonical: src.configs.policies.other.reinforcement_learning.RewardShapingConfig.rejected_reward
:type: float
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RewardShapingConfig.rejected_reward
```

````

````{py:attribute} stagnation_penalty
:canonical: src.configs.policies.other.reinforcement_learning.RewardShapingConfig.stagnation_penalty
:type: float
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RewardShapingConfig.stagnation_penalty
```

````

````{py:attribute} adaptive_rewards
:canonical: src.configs.policies.other.reinforcement_learning.RewardShapingConfig.adaptive_rewards
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RewardShapingConfig.adaptive_rewards
```

````

````{py:attribute} normalize_rewards
:canonical: src.configs.policies.other.reinforcement_learning.RewardShapingConfig.normalize_rewards
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RewardShapingConfig.normalize_rewards
```

````

````{py:attribute} improvement_threshold
:canonical: src.configs.policies.other.reinforcement_learning.RewardShapingConfig.improvement_threshold
:type: float
:value: >
   1e-06

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RewardShapingConfig.improvement_threshold
```

````

````{py:attribute} rewards_size
:canonical: src.configs.policies.other.reinforcement_learning.RewardShapingConfig.rewards_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RewardShapingConfig.rewards_size
```

````

`````

`````{py:class} FeatureExtractorConfig
:canonical: src.configs.policies.other.reinforcement_learning.FeatureExtractorConfig

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.FeatureExtractorConfig
```

````{py:attribute} progress_thresholds
:canonical: src.configs.policies.other.reinforcement_learning.FeatureExtractorConfig.progress_thresholds
:type: typing.List[float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.FeatureExtractorConfig.progress_thresholds
```

````

````{py:attribute} stagnation_thresholds
:canonical: src.configs.policies.other.reinforcement_learning.FeatureExtractorConfig.stagnation_thresholds
:type: typing.List[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.FeatureExtractorConfig.stagnation_thresholds
```

````

````{py:attribute} diversity_thresholds
:canonical: src.configs.policies.other.reinforcement_learning.FeatureExtractorConfig.diversity_thresholds
:type: typing.List[float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.FeatureExtractorConfig.diversity_thresholds
```

````

````{py:attribute} diversity_history_size
:canonical: src.configs.policies.other.reinforcement_learning.FeatureExtractorConfig.diversity_history_size
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.FeatureExtractorConfig.diversity_history_size
```

````

````{py:attribute} improvement_history_size
:canonical: src.configs.policies.other.reinforcement_learning.FeatureExtractorConfig.improvement_history_size
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.FeatureExtractorConfig.improvement_history_size
```

````

`````

`````{py:class} ContextFeatureExtractorConfig
:canonical: src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig
```

````{py:attribute} alpha
:canonical: src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.alpha
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.alpha
```

````

````{py:attribute} feature_dim
:canonical: src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.feature_dim
:type: int
:value: >
   8

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.feature_dim
```

````

````{py:attribute} selection_threshold
:canonical: src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.selection_threshold
:type: float
:value: >
   1e-09

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.selection_threshold
```

````

````{py:attribute} lambda_prior
:canonical: src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.lambda_prior
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.lambda_prior
```

````

````{py:attribute} noise_variance
:canonical: src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.noise_variance
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.noise_variance
```

````

````{py:attribute} epsilon
:canonical: src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.epsilon
:type: float
:value: >
   0.15

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.epsilon
```

````

````{py:attribute} epsilon_decay
:canonical: src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.epsilon_decay
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.epsilon_decay
```

````

````{py:attribute} epsilon_decay_step
:canonical: src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.epsilon_decay_step
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.epsilon_decay_step
```

````

````{py:attribute} epsilon_min
:canonical: src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.epsilon_min
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig.epsilon_min
```

````

`````

`````{py:class} GPCMABConfig
:canonical: src.configs.policies.other.reinforcement_learning.GPCMABConfig

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.GPCMABConfig
```

````{py:attribute} beta
:canonical: src.configs.policies.other.reinforcement_learning.GPCMABConfig.beta
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.GPCMABConfig.beta
```

````

````{py:attribute} length_scale
:canonical: src.configs.policies.other.reinforcement_learning.GPCMABConfig.length_scale
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.GPCMABConfig.length_scale
```

````

````{py:attribute} signal_variance
:canonical: src.configs.policies.other.reinforcement_learning.GPCMABConfig.signal_variance
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.GPCMABConfig.signal_variance
```

````

````{py:attribute} noise_variance
:canonical: src.configs.policies.other.reinforcement_learning.GPCMABConfig.noise_variance
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.GPCMABConfig.noise_variance
```

````

````{py:attribute} max_history
:canonical: src.configs.policies.other.reinforcement_learning.GPCMABConfig.max_history
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.GPCMABConfig.max_history
```

````

````{py:attribute} super_arm_size
:canonical: src.configs.policies.other.reinforcement_learning.GPCMABConfig.super_arm_size
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.GPCMABConfig.super_arm_size
```

````

`````

`````{py:class} RLConfig
:canonical: src.configs.policies.other.reinforcement_learning.RLConfig

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RLConfig
```

````{py:attribute} agent_type
:canonical: src.configs.policies.other.reinforcement_learning.RLConfig.agent_type
:type: str
:value: >
   'bandit'

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RLConfig.agent_type
```

````

````{py:attribute} bandit
:canonical: src.configs.policies.other.reinforcement_learning.RLConfig.bandit
:type: src.configs.policies.other.reinforcement_learning.BanditConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RLConfig.bandit
```

````

````{py:attribute} td_learning
:canonical: src.configs.policies.other.reinforcement_learning.RLConfig.td_learning
:type: src.configs.policies.other.reinforcement_learning.TDLearningConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RLConfig.td_learning
```

````

````{py:attribute} sarsa
:canonical: src.configs.policies.other.reinforcement_learning.RLConfig.sarsa
:type: typing.Optional[src.configs.policies.other.reinforcement_learning.TDLearningConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RLConfig.sarsa
```

````

````{py:attribute} contextual
:canonical: src.configs.policies.other.reinforcement_learning.RLConfig.contextual
:type: src.configs.policies.other.reinforcement_learning.LinUCBConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RLConfig.contextual
```

````

````{py:attribute} gp_cmab
:canonical: src.configs.policies.other.reinforcement_learning.RLConfig.gp_cmab
:type: src.configs.policies.other.reinforcement_learning.GPCMABConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RLConfig.gp_cmab
```

````

````{py:attribute} evolution_cmab
:canonical: src.configs.policies.other.reinforcement_learning.RLConfig.evolution_cmab
:type: src.configs.policies.other.reinforcement_learning.EvolutionaryCMABConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RLConfig.evolution_cmab
```

````

````{py:attribute} reward
:canonical: src.configs.policies.other.reinforcement_learning.RLConfig.reward
:type: src.configs.policies.other.reinforcement_learning.RewardShapingConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RLConfig.reward
```

````

````{py:attribute} features
:canonical: src.configs.policies.other.reinforcement_learning.RLConfig.features
:type: src.configs.policies.other.reinforcement_learning.FeatureExtractorConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RLConfig.features
```

````

````{py:attribute} context_features
:canonical: src.configs.policies.other.reinforcement_learning.RLConfig.context_features
:type: src.configs.policies.other.reinforcement_learning.ContextFeatureExtractorConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RLConfig.context_features
```

````

````{py:attribute} params
:canonical: src.configs.policies.other.reinforcement_learning.RLConfig.params
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.reinforcement_learning.RLConfig.params
```

````

`````
