# {py:mod}`src.configs.tasks.meta_rl`

```{py:module} src.configs.tasks.meta_rl
```

```{autodoc2-docstring} src.configs.tasks.meta_rl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MetaRLConfig <src.configs.tasks.meta_rl.MetaRLConfig>`
  - ```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig
    :summary:
    ```
````

### API

`````{py:class} MetaRLConfig
:canonical: src.configs.tasks.meta_rl.MetaRLConfig

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig
```

````{py:attribute} use_meta
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.use_meta
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.use_meta
```

````

````{py:attribute} meta_strategy
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.meta_strategy
:type: str
:value: >
   'rnn'

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.meta_strategy
```

````

````{py:attribute} meta_lr
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.meta_lr
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.meta_lr
```

````

````{py:attribute} meta_hidden_dim
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.meta_hidden_dim
:type: int
:value: >
   64

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.meta_hidden_dim
```

````

````{py:attribute} meta_history_length
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.meta_history_length
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.meta_history_length
```

````

````{py:attribute} mrl_exploration_factor
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.mrl_exploration_factor
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.mrl_exploration_factor
```

````

````{py:attribute} mrl_range
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.mrl_range
:type: typing.List[float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.mrl_range
```

````

````{py:attribute} mrl_batch_size
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.mrl_batch_size
:type: int
:value: >
   256

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.mrl_batch_size
```

````

````{py:attribute} mrl_step
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.mrl_step
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.mrl_step
```

````

````{py:attribute} hrl_threshold
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.hrl_threshold
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.hrl_threshold
```

````

````{py:attribute} hrl_epochs
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.hrl_epochs
:type: int
:value: >
   4

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.hrl_epochs
```

````

````{py:attribute} hrl_clip_eps
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.hrl_clip_eps
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.hrl_clip_eps
```

````

````{py:attribute} hrl_pid_target
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.hrl_pid_target
:type: float
:value: >
   0.0003

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.hrl_pid_target
```

````

````{py:attribute} hrl_lambda_waste
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.hrl_lambda_waste
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.hrl_lambda_waste
```

````

````{py:attribute} hrl_lambda_cost
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.hrl_lambda_cost
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.hrl_lambda_cost
```

````

````{py:attribute} hrl_lambda_overflow_initial
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.hrl_lambda_overflow_initial
:type: float
:value: >
   2000.0

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.hrl_lambda_overflow_initial
```

````

````{py:attribute} hrl_lambda_overflow_min
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.hrl_lambda_overflow_min
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.hrl_lambda_overflow_min
```

````

````{py:attribute} hrl_lambda_overflow_max
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.hrl_lambda_overflow_max
:type: float
:value: >
   5000.0

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.hrl_lambda_overflow_max
```

````

````{py:attribute} hrl_lambda_pruning
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.hrl_lambda_pruning
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.hrl_lambda_pruning
```

````

````{py:attribute} hrl_lambda_mask_aux
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.hrl_lambda_mask_aux
:type: float
:value: >
   5.0

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.hrl_lambda_mask_aux
```

````

````{py:attribute} hrl_entropy_coef
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.hrl_entropy_coef
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.hrl_entropy_coef
```

````

````{py:attribute} shared_encoder
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.shared_encoder
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.shared_encoder
```

````

````{py:attribute} gat_hidden_dim
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.gat_hidden_dim
:type: int
:value: >
   128

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.gat_hidden_dim
```

````

````{py:attribute} lstm_hidden_dim
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.lstm_hidden_dim
:type: int
:value: >
   64

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.lstm_hidden_dim
```

````

````{py:attribute} gate_prob_threshold
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.gate_prob_threshold
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.gate_prob_threshold
```

````

````{py:attribute} lr_critic_value
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.lr_critic_value
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.lr_critic_value
```

````

````{py:attribute} cb_exploration_method
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.cb_exploration_method
:type: str
:value: >
   'ucb'

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.cb_exploration_method
```

````

````{py:attribute} cb_num_configs
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.cb_num_configs
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.cb_num_configs
```

````

````{py:attribute} cb_epsilon_decay
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.cb_epsilon_decay
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.cb_epsilon_decay
```

````

````{py:attribute} cb_min_epsilon
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.cb_min_epsilon
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.cb_min_epsilon
```

````

````{py:attribute} cb_context_features
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.cb_context_features
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.cb_context_features
```

````

````{py:attribute} cb_features_aggregation
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.cb_features_aggregation
:type: str
:value: >
   'avg'

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.cb_features_aggregation
```

````

````{py:attribute} morl_objectives
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.morl_objectives
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.morl_objectives
```

````

````{py:attribute} morl_adaptation_rate
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.morl_adaptation_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.morl_adaptation_rate
```

````

````{py:attribute} graph
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.graph
:type: src.configs.envs.graph.GraphConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.graph
```

````

````{py:attribute} reward
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.reward
:type: src.configs.envs.objective.ObjectiveConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.reward
```

````

`````
