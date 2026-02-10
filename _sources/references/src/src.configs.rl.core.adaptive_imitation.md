# {py:mod}`src.configs.rl.core.adaptive_imitation`

```{py:module} src.configs.rl.core.adaptive_imitation
```

```{autodoc2-docstring} src.configs.rl.core.adaptive_imitation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AdaptiveImitationConfig <src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig>`
  - ```{autodoc2-docstring} src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ExpertPolicyConfig <src.configs.rl.core.adaptive_imitation.ExpertPolicyConfig>`
  - ```{autodoc2-docstring} src.configs.rl.core.adaptive_imitation.ExpertPolicyConfig
    :summary:
    ```
````

### API

````{py:data} ExpertPolicyConfig
:canonical: src.configs.rl.core.adaptive_imitation.ExpertPolicyConfig
:value: >
   None

```{autodoc2-docstring} src.configs.rl.core.adaptive_imitation.ExpertPolicyConfig
```

````

`````{py:class} AdaptiveImitationConfig
:canonical: src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig

```{autodoc2-docstring} src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig
```

````{py:attribute} il_weight
:canonical: src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.il_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.il_weight
```

````

````{py:attribute} il_decay
:canonical: src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.il_decay
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.il_decay
```

````

````{py:attribute} patience
:canonical: src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.patience
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.patience
```

````

````{py:attribute} threshold
:canonical: src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.threshold
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.threshold
```

````

````{py:attribute} decay_step
:canonical: src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.decay_step
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.decay_step
```

````

````{py:attribute} epsilon
:canonical: src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.epsilon
:type: float
:value: >
   1e-05

```{autodoc2-docstring} src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.epsilon
```

````

````{py:attribute} policy_config
:canonical: src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.policy_config
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.policy_config
```

````

````{py:attribute} loss_fn
:canonical: src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.loss_fn
:type: str
:value: >
   'nll'

```{autodoc2-docstring} src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.loss_fn
```

````

````{py:method} __post_init__()
:canonical: src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.__post_init__

```{autodoc2-docstring} src.configs.rl.core.adaptive_imitation.AdaptiveImitationConfig.__post_init__
```

````

`````
