# {py:mod}`src.configs.core.rl`

```{py:module} src.configs.core.rl
```

```{autodoc2-docstring} src.configs.core.rl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLConfig <src.configs.core.rl.RLConfig>`
  - ```{autodoc2-docstring} src.configs.core.rl.RLConfig
    :summary:
    ```
````

### API

`````{py:class} RLConfig
:canonical: src.configs.core.rl.RLConfig

```{autodoc2-docstring} src.configs.core.rl.RLConfig
```

````{py:attribute} algorithm
:canonical: src.configs.core.rl.RLConfig.algorithm
:type: str
:value: >
   'reinforce'

```{autodoc2-docstring} src.configs.core.rl.RLConfig.algorithm
```

````

````{py:attribute} baseline
:canonical: src.configs.core.rl.RLConfig.baseline
:type: str
:value: >
   'rollout'

```{autodoc2-docstring} src.configs.core.rl.RLConfig.baseline
```

````

````{py:attribute} bl_warmup_epochs
:canonical: src.configs.core.rl.RLConfig.bl_warmup_epochs
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.core.rl.RLConfig.bl_warmup_epochs
```

````

````{py:attribute} entropy_weight
:canonical: src.configs.core.rl.RLConfig.entropy_weight
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.core.rl.RLConfig.entropy_weight
```

````

````{py:attribute} max_grad_norm
:canonical: src.configs.core.rl.RLConfig.max_grad_norm
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.core.rl.RLConfig.max_grad_norm
```

````

````{py:attribute} gamma
:canonical: src.configs.core.rl.RLConfig.gamma
:type: float
:value: >
   0.99

```{autodoc2-docstring} src.configs.core.rl.RLConfig.gamma
```

````

````{py:attribute} exp_beta
:canonical: src.configs.core.rl.RLConfig.exp_beta
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.configs.core.rl.RLConfig.exp_beta
```

````

````{py:attribute} bl_alpha
:canonical: src.configs.core.rl.RLConfig.bl_alpha
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.core.rl.RLConfig.bl_alpha
```

````

````{py:attribute} ppo
:canonical: src.configs.core.rl.RLConfig.ppo
:type: src.configs.core.ppo.PPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.core.rl.RLConfig.ppo
```

````

````{py:attribute} sapo
:canonical: src.configs.core.rl.RLConfig.sapo
:type: src.configs.core.sapo.SAPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.core.rl.RLConfig.sapo
```

````

````{py:attribute} grpo
:canonical: src.configs.core.rl.RLConfig.grpo
:type: src.configs.core.grpo.GRPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.core.rl.RLConfig.grpo
```

````

````{py:attribute} pomo
:canonical: src.configs.core.rl.RLConfig.pomo
:type: src.configs.core.pomo.POMOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.core.rl.RLConfig.pomo
```

````

````{py:attribute} symnco
:canonical: src.configs.core.rl.RLConfig.symnco
:type: src.configs.core.symnco.SymNCOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.core.rl.RLConfig.symnco
```

````

````{py:attribute} imitation
:canonical: src.configs.core.rl.RLConfig.imitation
:type: src.configs.core.imitation.ImitationConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.core.rl.RLConfig.imitation
```

````

````{py:attribute} gdpo
:canonical: src.configs.core.rl.RLConfig.gdpo
:type: src.configs.core.gdpo.GDPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.core.rl.RLConfig.gdpo
```

````

````{py:attribute} adaptive_imitation
:canonical: src.configs.core.rl.RLConfig.adaptive_imitation
:type: src.configs.core.adaptive_imitation.AdaptiveImitationConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.core.rl.RLConfig.adaptive_imitation
```

````

`````
