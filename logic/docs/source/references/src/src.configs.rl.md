# {py:mod}`src.configs.rl`

```{py:module} src.configs.rl
```

```{autodoc2-docstring} src.configs.rl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLConfig <src.configs.rl.RLConfig>`
  - ```{autodoc2-docstring} src.configs.rl.RLConfig
    :summary:
    ```
````

### API

`````{py:class} RLConfig
:canonical: src.configs.rl.RLConfig

```{autodoc2-docstring} src.configs.rl.RLConfig
```

````{py:attribute} algorithm
:canonical: src.configs.rl.RLConfig.algorithm
:type: str
:value: >
   'reinforce'

```{autodoc2-docstring} src.configs.rl.RLConfig.algorithm
```

````

````{py:attribute} baseline
:canonical: src.configs.rl.RLConfig.baseline
:type: str
:value: >
   'rollout'

```{autodoc2-docstring} src.configs.rl.RLConfig.baseline
```

````

````{py:attribute} bl_warmup_epochs
:canonical: src.configs.rl.RLConfig.bl_warmup_epochs
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.rl.RLConfig.bl_warmup_epochs
```

````

````{py:attribute} entropy_weight
:canonical: src.configs.rl.RLConfig.entropy_weight
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.rl.RLConfig.entropy_weight
```

````

````{py:attribute} max_grad_norm
:canonical: src.configs.rl.RLConfig.max_grad_norm
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.rl.RLConfig.max_grad_norm
```

````

````{py:attribute} ppo_epochs
:canonical: src.configs.rl.RLConfig.ppo_epochs
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.rl.RLConfig.ppo_epochs
```

````

````{py:attribute} eps_clip
:canonical: src.configs.rl.RLConfig.eps_clip
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.rl.RLConfig.eps_clip
```

````

````{py:attribute} value_loss_weight
:canonical: src.configs.rl.RLConfig.value_loss_weight
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.rl.RLConfig.value_loss_weight
```

````

````{py:attribute} mini_batch_size
:canonical: src.configs.rl.RLConfig.mini_batch_size
:type: float
:value: >
   0.25

```{autodoc2-docstring} src.configs.rl.RLConfig.mini_batch_size
```

````

````{py:attribute} sapo_tau_pos
:canonical: src.configs.rl.RLConfig.sapo_tau_pos
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.rl.RLConfig.sapo_tau_pos
```

````

````{py:attribute} sapo_tau_neg
:canonical: src.configs.rl.RLConfig.sapo_tau_neg
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.rl.RLConfig.sapo_tau_neg
```

````

````{py:attribute} dr_grpo_group_size
:canonical: src.configs.rl.RLConfig.dr_grpo_group_size
:type: int
:value: >
   8

```{autodoc2-docstring} src.configs.rl.RLConfig.dr_grpo_group_size
```

````

````{py:attribute} dr_grpo_epsilon
:canonical: src.configs.rl.RLConfig.dr_grpo_epsilon
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.rl.RLConfig.dr_grpo_epsilon
```

````

````{py:attribute} num_augment
:canonical: src.configs.rl.RLConfig.num_augment
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.rl.RLConfig.num_augment
```

````

````{py:attribute} num_starts
:canonical: src.configs.rl.RLConfig.num_starts
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.rl.RLConfig.num_starts
```

````

````{py:attribute} augment_fn
:canonical: src.configs.rl.RLConfig.augment_fn
:type: str
:value: >
   'dihedral8'

```{autodoc2-docstring} src.configs.rl.RLConfig.augment_fn
```

````

````{py:attribute} symnco_alpha
:canonical: src.configs.rl.RLConfig.symnco_alpha
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.rl.RLConfig.symnco_alpha
```

````

````{py:attribute} symnco_beta
:canonical: src.configs.rl.RLConfig.symnco_beta
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.rl.RLConfig.symnco_beta
```

````

````{py:attribute} imitation_mode
:canonical: src.configs.rl.RLConfig.imitation_mode
:type: str
:value: >
   'hgs'

```{autodoc2-docstring} src.configs.rl.RLConfig.imitation_mode
```

````

````{py:attribute} imitation_weight
:canonical: src.configs.rl.RLConfig.imitation_weight
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.rl.RLConfig.imitation_weight
```

````

````{py:attribute} imitation_decay
:canonical: src.configs.rl.RLConfig.imitation_decay
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.rl.RLConfig.imitation_decay
```

````

````{py:attribute} imitation_threshold
:canonical: src.configs.rl.RLConfig.imitation_threshold
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.rl.RLConfig.imitation_threshold
```

````

````{py:attribute} reannealing_threshold
:canonical: src.configs.rl.RLConfig.reannealing_threshold
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.rl.RLConfig.reannealing_threshold
```

````

````{py:attribute} reannealing_patience
:canonical: src.configs.rl.RLConfig.reannealing_patience
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.rl.RLConfig.reannealing_patience
```

````

````{py:attribute} random_ls_iterations
:canonical: src.configs.rl.RLConfig.random_ls_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.rl.RLConfig.random_ls_iterations
```

````

````{py:attribute} random_ls_op_probs
:canonical: src.configs.rl.RLConfig.random_ls_op_probs
:type: typing.Optional[typing.Dict[str, float]]
:value: >
   None

```{autodoc2-docstring} src.configs.rl.RLConfig.random_ls_op_probs
```

````

````{py:attribute} gdpo_objective_keys
:canonical: src.configs.rl.RLConfig.gdpo_objective_keys
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.gdpo_objective_keys
```

````

````{py:attribute} gdpo_objective_weights
:canonical: src.configs.rl.RLConfig.gdpo_objective_weights
:type: typing.Optional[typing.List[float]]
:value: >
   None

```{autodoc2-docstring} src.configs.rl.RLConfig.gdpo_objective_weights
```

````

````{py:attribute} gdpo_conditional_key
:canonical: src.configs.rl.RLConfig.gdpo_conditional_key
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.rl.RLConfig.gdpo_conditional_key
```

````

````{py:attribute} gdpo_renormalize
:canonical: src.configs.rl.RLConfig.gdpo_renormalize
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.rl.RLConfig.gdpo_renormalize
```

````

````{py:attribute} gamma
:canonical: src.configs.rl.RLConfig.gamma
:type: float
:value: >
   0.99

```{autodoc2-docstring} src.configs.rl.RLConfig.gamma
```

````

````{py:attribute} gspo_epsilon
:canonical: src.configs.rl.RLConfig.gspo_epsilon
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.rl.RLConfig.gspo_epsilon
```

````

````{py:attribute} gspo_epochs
:canonical: src.configs.rl.RLConfig.gspo_epochs
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.rl.RLConfig.gspo_epochs
```

````

````{py:attribute} dr_grpo_epochs
:canonical: src.configs.rl.RLConfig.dr_grpo_epochs
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.rl.RLConfig.dr_grpo_epochs
```

````

````{py:attribute} exp_beta
:canonical: src.configs.rl.RLConfig.exp_beta
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.configs.rl.RLConfig.exp_beta
```

````

````{py:attribute} bl_alpha
:canonical: src.configs.rl.RLConfig.bl_alpha
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.rl.RLConfig.bl_alpha
```

````

````{py:attribute} imitation_decay_step
:canonical: src.configs.rl.RLConfig.imitation_decay_step
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.rl.RLConfig.imitation_decay_step
```

````

`````
