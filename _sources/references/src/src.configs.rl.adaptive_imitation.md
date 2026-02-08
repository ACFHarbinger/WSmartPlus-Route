# {py:mod}`src.configs.rl.adaptive_imitation`

```{py:module} src.configs.rl.adaptive_imitation
```

```{autodoc2-docstring} src.configs.rl.adaptive_imitation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AdaptiveImitationConfig <src.configs.rl.adaptive_imitation.AdaptiveImitationConfig>`
  - ```{autodoc2-docstring} src.configs.rl.adaptive_imitation.AdaptiveImitationConfig
    :summary:
    ```
````

### API

`````{py:class} AdaptiveImitationConfig
:canonical: src.configs.rl.adaptive_imitation.AdaptiveImitationConfig

```{autodoc2-docstring} src.configs.rl.adaptive_imitation.AdaptiveImitationConfig
```

````{py:attribute} il_weight
:canonical: src.configs.rl.adaptive_imitation.AdaptiveImitationConfig.il_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.rl.adaptive_imitation.AdaptiveImitationConfig.il_weight
```

````

````{py:attribute} il_decay
:canonical: src.configs.rl.adaptive_imitation.AdaptiveImitationConfig.il_decay
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.rl.adaptive_imitation.AdaptiveImitationConfig.il_decay
```

````

````{py:attribute} patience
:canonical: src.configs.rl.adaptive_imitation.AdaptiveImitationConfig.patience
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.rl.adaptive_imitation.AdaptiveImitationConfig.patience
```

````

````{py:attribute} threshold
:canonical: src.configs.rl.adaptive_imitation.AdaptiveImitationConfig.threshold
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.rl.adaptive_imitation.AdaptiveImitationConfig.threshold
```

````

````{py:attribute} decay_step
:canonical: src.configs.rl.adaptive_imitation.AdaptiveImitationConfig.decay_step
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.rl.adaptive_imitation.AdaptiveImitationConfig.decay_step
```

````

````{py:attribute} epsilon
:canonical: src.configs.rl.adaptive_imitation.AdaptiveImitationConfig.epsilon
:type: float
:value: >
   1e-05

```{autodoc2-docstring} src.configs.rl.adaptive_imitation.AdaptiveImitationConfig.epsilon
```

````

`````
