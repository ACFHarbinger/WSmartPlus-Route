# {py:mod}`src.models.l2d.ppo_policy`

```{py:module} src.models.l2d.ppo_policy
```

```{autodoc2-docstring} src.models.l2d.ppo_policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`L2DPolicy4PPO <src.models.l2d.ppo_policy.L2DPolicy4PPO>`
  - ```{autodoc2-docstring} src.models.l2d.ppo_policy.L2DPolicy4PPO
    :summary:
    ```
````

### API

````{py:class} L2DPolicy4PPO(embed_dim: int = 128, num_encoder_layers: int = 3, feedforward_hidden: int = 512, env_name: str = 'jssp', temp: float = 1.0, tanh_clipping: float = 10.0, train_decode_type: str = 'sampling', val_decode_type: str = 'greedy', test_decode_type: str = 'greedy', **kwargs)
:canonical: src.models.l2d.ppo_policy.L2DPolicy4PPO

Bases: {py:obj}`src.models.l2d.policy.L2DPolicy`

```{autodoc2-docstring} src.models.l2d.ppo_policy.L2DPolicy4PPO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.l2d.ppo_policy.L2DPolicy4PPO.__init__
```

````
