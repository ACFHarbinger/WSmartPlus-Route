# {py:mod}`src.configs.rl.core.dr_alns`

```{py:module} src.configs.rl.core.dr_alns
```

```{autodoc2-docstring} src.configs.rl.core.dr_alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DRALNSConfig <src.configs.rl.core.dr_alns.DRALNSConfig>`
  - ```{autodoc2-docstring} src.configs.rl.core.dr_alns.DRALNSConfig
    :summary:
    ```
````

### API

`````{py:class} DRALNSConfig
:canonical: src.configs.rl.core.dr_alns.DRALNSConfig

```{autodoc2-docstring} src.configs.rl.core.dr_alns.DRALNSConfig
```

````{py:attribute} max_iterations
:canonical: src.configs.rl.core.dr_alns.DRALNSConfig.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.rl.core.dr_alns.DRALNSConfig.max_iterations
```

````

````{py:attribute} n_destroy_ops
:canonical: src.configs.rl.core.dr_alns.DRALNSConfig.n_destroy_ops
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.rl.core.dr_alns.DRALNSConfig.n_destroy_ops
```

````

````{py:attribute} n_repair_ops
:canonical: src.configs.rl.core.dr_alns.DRALNSConfig.n_repair_ops
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.rl.core.dr_alns.DRALNSConfig.n_repair_ops
```

````

````{py:attribute} n_steps_per_epoch
:canonical: src.configs.rl.core.dr_alns.DRALNSConfig.n_steps_per_epoch
:type: int
:value: >
   2048

```{autodoc2-docstring} src.configs.rl.core.dr_alns.DRALNSConfig.n_steps_per_epoch
```

````

````{py:attribute} batch_size
:canonical: src.configs.rl.core.dr_alns.DRALNSConfig.batch_size
:type: int
:value: >
   64

```{autodoc2-docstring} src.configs.rl.core.dr_alns.DRALNSConfig.batch_size
```

````

````{py:attribute} n_epochs
:canonical: src.configs.rl.core.dr_alns.DRALNSConfig.n_epochs
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.rl.core.dr_alns.DRALNSConfig.n_epochs
```

````

````{py:attribute} lr
:canonical: src.configs.rl.core.dr_alns.DRALNSConfig.lr
:type: float
:value: >
   0.0003

```{autodoc2-docstring} src.configs.rl.core.dr_alns.DRALNSConfig.lr
```

````

````{py:attribute} gamma
:canonical: src.configs.rl.core.dr_alns.DRALNSConfig.gamma
:type: float
:value: >
   0.99

```{autodoc2-docstring} src.configs.rl.core.dr_alns.DRALNSConfig.gamma
```

````

````{py:attribute} gae_lambda
:canonical: src.configs.rl.core.dr_alns.DRALNSConfig.gae_lambda
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.rl.core.dr_alns.DRALNSConfig.gae_lambda
```

````

````{py:attribute} clip_epsilon
:canonical: src.configs.rl.core.dr_alns.DRALNSConfig.clip_epsilon
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.rl.core.dr_alns.DRALNSConfig.clip_epsilon
```

````

````{py:attribute} value_loss_coef
:canonical: src.configs.rl.core.dr_alns.DRALNSConfig.value_loss_coef
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.rl.core.dr_alns.DRALNSConfig.value_loss_coef
```

````

````{py:attribute} entropy_coef
:canonical: src.configs.rl.core.dr_alns.DRALNSConfig.entropy_coef
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.rl.core.dr_alns.DRALNSConfig.entropy_coef
```

````

````{py:attribute} max_grad_norm
:canonical: src.configs.rl.core.dr_alns.DRALNSConfig.max_grad_norm
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.rl.core.dr_alns.DRALNSConfig.max_grad_norm
```

````

`````
