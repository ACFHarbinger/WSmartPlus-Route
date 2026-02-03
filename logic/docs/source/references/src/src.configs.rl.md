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

* - {py:obj}`PPOConfig <src.configs.rl.PPOConfig>`
  - ```{autodoc2-docstring} src.configs.rl.PPOConfig
    :summary:
    ```
* - {py:obj}`SAPOConfig <src.configs.rl.SAPOConfig>`
  - ```{autodoc2-docstring} src.configs.rl.SAPOConfig
    :summary:
    ```
* - {py:obj}`GRPOConfig <src.configs.rl.GRPOConfig>`
  - ```{autodoc2-docstring} src.configs.rl.GRPOConfig
    :summary:
    ```
* - {py:obj}`POMOConfig <src.configs.rl.POMOConfig>`
  - ```{autodoc2-docstring} src.configs.rl.POMOConfig
    :summary:
    ```
* - {py:obj}`SymNCOConfig <src.configs.rl.SymNCOConfig>`
  - ```{autodoc2-docstring} src.configs.rl.SymNCOConfig
    :summary:
    ```
* - {py:obj}`ImitationConfig <src.configs.rl.ImitationConfig>`
  - ```{autodoc2-docstring} src.configs.rl.ImitationConfig
    :summary:
    ```
* - {py:obj}`GDPOConfig <src.configs.rl.GDPOConfig>`
  - ```{autodoc2-docstring} src.configs.rl.GDPOConfig
    :summary:
    ```
* - {py:obj}`AdaptiveImitationConfig <src.configs.rl.AdaptiveImitationConfig>`
  - ```{autodoc2-docstring} src.configs.rl.AdaptiveImitationConfig
    :summary:
    ```
* - {py:obj}`RLConfig <src.configs.rl.RLConfig>`
  - ```{autodoc2-docstring} src.configs.rl.RLConfig
    :summary:
    ```
````

### API

`````{py:class} PPOConfig
:canonical: src.configs.rl.PPOConfig

```{autodoc2-docstring} src.configs.rl.PPOConfig
```

````{py:attribute} epochs
:canonical: src.configs.rl.PPOConfig.epochs
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.rl.PPOConfig.epochs
```

````

````{py:attribute} eps_clip
:canonical: src.configs.rl.PPOConfig.eps_clip
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.rl.PPOConfig.eps_clip
```

````

````{py:attribute} value_loss_weight
:canonical: src.configs.rl.PPOConfig.value_loss_weight
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.rl.PPOConfig.value_loss_weight
```

````

````{py:attribute} mini_batch_size
:canonical: src.configs.rl.PPOConfig.mini_batch_size
:type: float
:value: >
   0.25

```{autodoc2-docstring} src.configs.rl.PPOConfig.mini_batch_size
```

````

`````

`````{py:class} SAPOConfig
:canonical: src.configs.rl.SAPOConfig

```{autodoc2-docstring} src.configs.rl.SAPOConfig
```

````{py:attribute} tau_pos
:canonical: src.configs.rl.SAPOConfig.tau_pos
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.rl.SAPOConfig.tau_pos
```

````

````{py:attribute} tau_neg
:canonical: src.configs.rl.SAPOConfig.tau_neg
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.rl.SAPOConfig.tau_neg
```

````

`````

`````{py:class} GRPOConfig
:canonical: src.configs.rl.GRPOConfig

```{autodoc2-docstring} src.configs.rl.GRPOConfig
```

````{py:attribute} group_size
:canonical: src.configs.rl.GRPOConfig.group_size
:type: int
:value: >
   8

```{autodoc2-docstring} src.configs.rl.GRPOConfig.group_size
```

````

````{py:attribute} epsilon
:canonical: src.configs.rl.GRPOConfig.epsilon
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.rl.GRPOConfig.epsilon
```

````

````{py:attribute} epochs
:canonical: src.configs.rl.GRPOConfig.epochs
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.rl.GRPOConfig.epochs
```

````

`````

`````{py:class} POMOConfig
:canonical: src.configs.rl.POMOConfig

```{autodoc2-docstring} src.configs.rl.POMOConfig
```

````{py:attribute} num_augment
:canonical: src.configs.rl.POMOConfig.num_augment
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.rl.POMOConfig.num_augment
```

````

````{py:attribute} num_starts
:canonical: src.configs.rl.POMOConfig.num_starts
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.rl.POMOConfig.num_starts
```

````

````{py:attribute} augment_fn
:canonical: src.configs.rl.POMOConfig.augment_fn
:type: str
:value: >
   'dihedral8'

```{autodoc2-docstring} src.configs.rl.POMOConfig.augment_fn
```

````

`````

`````{py:class} SymNCOConfig
:canonical: src.configs.rl.SymNCOConfig

```{autodoc2-docstring} src.configs.rl.SymNCOConfig
```

````{py:attribute} alpha
:canonical: src.configs.rl.SymNCOConfig.alpha
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.rl.SymNCOConfig.alpha
```

````

````{py:attribute} beta
:canonical: src.configs.rl.SymNCOConfig.beta
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.rl.SymNCOConfig.beta
```

````

`````

`````{py:class} ImitationConfig
:canonical: src.configs.rl.ImitationConfig

```{autodoc2-docstring} src.configs.rl.ImitationConfig
```

````{py:attribute} mode
:canonical: src.configs.rl.ImitationConfig.mode
:type: str
:value: >
   'hgs'

```{autodoc2-docstring} src.configs.rl.ImitationConfig.mode
```

````

````{py:attribute} weight
:canonical: src.configs.rl.ImitationConfig.weight
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.rl.ImitationConfig.weight
```

````

````{py:attribute} decay
:canonical: src.configs.rl.ImitationConfig.decay
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.rl.ImitationConfig.decay
```

````

````{py:attribute} threshold
:canonical: src.configs.rl.ImitationConfig.threshold
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.rl.ImitationConfig.threshold
```

````

````{py:attribute} decay_step
:canonical: src.configs.rl.ImitationConfig.decay_step
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.rl.ImitationConfig.decay_step
```

````

````{py:attribute} reannealing_threshold
:canonical: src.configs.rl.ImitationConfig.reannealing_threshold
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.rl.ImitationConfig.reannealing_threshold
```

````

````{py:attribute} reannealing_patience
:canonical: src.configs.rl.ImitationConfig.reannealing_patience
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.rl.ImitationConfig.reannealing_patience
```

````

````{py:attribute} random_ls_iterations
:canonical: src.configs.rl.ImitationConfig.random_ls_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.rl.ImitationConfig.random_ls_iterations
```

````

````{py:attribute} random_ls_op_probs
:canonical: src.configs.rl.ImitationConfig.random_ls_op_probs
:type: typing.Optional[typing.Dict[str, float]]
:value: >
   None

```{autodoc2-docstring} src.configs.rl.ImitationConfig.random_ls_op_probs
```

````

`````

`````{py:class} GDPOConfig
:canonical: src.configs.rl.GDPOConfig

```{autodoc2-docstring} src.configs.rl.GDPOConfig
```

````{py:attribute} objective_keys
:canonical: src.configs.rl.GDPOConfig.objective_keys
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.GDPOConfig.objective_keys
```

````

````{py:attribute} objective_weights
:canonical: src.configs.rl.GDPOConfig.objective_weights
:type: typing.Optional[typing.List[float]]
:value: >
   None

```{autodoc2-docstring} src.configs.rl.GDPOConfig.objective_weights
```

````

````{py:attribute} conditional_key
:canonical: src.configs.rl.GDPOConfig.conditional_key
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.rl.GDPOConfig.conditional_key
```

````

````{py:attribute} renormalize
:canonical: src.configs.rl.GDPOConfig.renormalize
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.rl.GDPOConfig.renormalize
```

````

`````

`````{py:class} AdaptiveImitationConfig
:canonical: src.configs.rl.AdaptiveImitationConfig

```{autodoc2-docstring} src.configs.rl.AdaptiveImitationConfig
```

````{py:attribute} il_weight
:canonical: src.configs.rl.AdaptiveImitationConfig.il_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.rl.AdaptiveImitationConfig.il_weight
```

````

````{py:attribute} il_decay
:canonical: src.configs.rl.AdaptiveImitationConfig.il_decay
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.rl.AdaptiveImitationConfig.il_decay
```

````

````{py:attribute} patience
:canonical: src.configs.rl.AdaptiveImitationConfig.patience
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.rl.AdaptiveImitationConfig.patience
```

````

`````

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

````{py:attribute} gamma
:canonical: src.configs.rl.RLConfig.gamma
:type: float
:value: >
   0.99

```{autodoc2-docstring} src.configs.rl.RLConfig.gamma
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

````{py:attribute} ppo
:canonical: src.configs.rl.RLConfig.ppo
:type: src.configs.rl.PPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.ppo
```

````

````{py:attribute} sapo
:canonical: src.configs.rl.RLConfig.sapo
:type: src.configs.rl.SAPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.sapo
```

````

````{py:attribute} grpo
:canonical: src.configs.rl.RLConfig.grpo
:type: src.configs.rl.GRPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.grpo
```

````

````{py:attribute} pomo
:canonical: src.configs.rl.RLConfig.pomo
:type: src.configs.rl.POMOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.pomo
```

````

````{py:attribute} symnco
:canonical: src.configs.rl.RLConfig.symnco
:type: src.configs.rl.SymNCOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.symnco
```

````

````{py:attribute} imitation
:canonical: src.configs.rl.RLConfig.imitation
:type: src.configs.rl.ImitationConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.imitation
```

````

````{py:attribute} gdpo
:canonical: src.configs.rl.RLConfig.gdpo
:type: src.configs.rl.GDPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.gdpo
```

````

````{py:attribute} adaptive_imitation
:canonical: src.configs.rl.RLConfig.adaptive_imitation
:type: src.configs.rl.AdaptiveImitationConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.adaptive_imitation
```

````

`````
