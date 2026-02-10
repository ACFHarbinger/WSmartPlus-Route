# {py:mod}`src.configs.rl`

```{py:module} src.configs.rl
```

```{autodoc2-docstring} src.configs.rl
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

src.configs.rl.policies
src.configs.rl.core
```

## Package Contents

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
:type: src.configs.rl.core.ppo.PPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.ppo
```

````

````{py:attribute} sapo
:canonical: src.configs.rl.RLConfig.sapo
:type: src.configs.rl.core.sapo.SAPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.sapo
```

````

````{py:attribute} grpo
:canonical: src.configs.rl.RLConfig.grpo
:type: src.configs.rl.core.grpo.GRPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.grpo
```

````

````{py:attribute} pomo
:canonical: src.configs.rl.RLConfig.pomo
:type: src.configs.rl.core.pomo.POMOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.pomo
```

````

````{py:attribute} symnco
:canonical: src.configs.rl.RLConfig.symnco
:type: src.configs.rl.core.symnco.SymNCOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.symnco
```

````

````{py:attribute} imitation
:canonical: src.configs.rl.RLConfig.imitation
:type: src.configs.rl.core.imitation.ImitationConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.imitation
```

````

````{py:attribute} gdpo
:canonical: src.configs.rl.RLConfig.gdpo
:type: src.configs.rl.core.gdpo.GDPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.gdpo
```

````

````{py:attribute} adaptive_imitation
:canonical: src.configs.rl.RLConfig.adaptive_imitation
:type: src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.rl.RLConfig.adaptive_imitation
```

````

`````
