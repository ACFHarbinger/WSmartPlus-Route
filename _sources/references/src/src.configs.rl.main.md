# {py:mod}`src.configs.rl.main`

```{py:module} src.configs.rl.main
```

```{autodoc2-docstring} src.configs.rl.main
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLConfig <src.configs.rl.main.RLConfig>`
  - ```{autodoc2-docstring} src.configs.rl.main.RLConfig
    :summary:
    ```
````

### API

`````{py:class} RLConfig
:canonical: src.configs.rl.main.RLConfig

```{autodoc2-docstring} src.configs.rl.main.RLConfig
```

````{py:attribute} algorithm
:canonical: src.configs.rl.main.RLConfig.algorithm
:type: str
:value: >
   'reinforce'

```{autodoc2-docstring} src.configs.rl.main.RLConfig.algorithm
```

````

````{py:attribute} baseline
:canonical: src.configs.rl.main.RLConfig.baseline
:type: str
:value: >
   'rollout'

```{autodoc2-docstring} src.configs.rl.main.RLConfig.baseline
```

````

````{py:attribute} bl_warmup_epochs
:canonical: src.configs.rl.main.RLConfig.bl_warmup_epochs
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.rl.main.RLConfig.bl_warmup_epochs
```

````

````{py:attribute} entropy_weight
:canonical: src.configs.rl.main.RLConfig.entropy_weight
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.rl.main.RLConfig.entropy_weight
```

````

````{py:attribute} max_grad_norm
:canonical: src.configs.rl.main.RLConfig.max_grad_norm
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.rl.main.RLConfig.max_grad_norm
```

````

````{py:attribute} gamma
:canonical: src.configs.rl.main.RLConfig.gamma
:type: float
:value: >
   0.99

```{autodoc2-docstring} src.configs.rl.main.RLConfig.gamma
```

````

````{py:attribute} exp_beta
:canonical: src.configs.rl.main.RLConfig.exp_beta
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.configs.rl.main.RLConfig.exp_beta
```

````

````{py:attribute} bl_alpha
:canonical: src.configs.rl.main.RLConfig.bl_alpha
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.rl.main.RLConfig.bl_alpha
```

````

````{py:attribute} ppo
:canonical: src.configs.rl.main.RLConfig.ppo
:type: src.configs.rl.ppo.PPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.main.RLConfig.ppo
```

````

````{py:attribute} sapo
:canonical: src.configs.rl.main.RLConfig.sapo
:type: src.configs.rl.sapo.SAPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.main.RLConfig.sapo
```

````

````{py:attribute} grpo
:canonical: src.configs.rl.main.RLConfig.grpo
:type: src.configs.rl.grpo.GRPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.main.RLConfig.grpo
```

````

````{py:attribute} pomo
:canonical: src.configs.rl.main.RLConfig.pomo
:type: src.configs.rl.pomo.POMOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.main.RLConfig.pomo
```

````

````{py:attribute} symnco
:canonical: src.configs.rl.main.RLConfig.symnco
:type: src.configs.rl.symnco.SymNCOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.main.RLConfig.symnco
```

````

````{py:attribute} imitation
:canonical: src.configs.rl.main.RLConfig.imitation
:type: src.configs.rl.imitation.ImitationConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.main.RLConfig.imitation
```

````

````{py:attribute} gdpo
:canonical: src.configs.rl.main.RLConfig.gdpo
:type: src.configs.rl.gdpo.GDPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.main.RLConfig.gdpo
```

````

````{py:attribute} adaptive_imitation
:canonical: src.configs.rl.main.RLConfig.adaptive_imitation
:type: src.configs.rl.adaptive_imitation.AdaptiveImitationConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.main.RLConfig.adaptive_imitation
```

````

`````
