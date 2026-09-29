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

````{py:attribute} shared_encoder
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.shared_encoder
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.shared_encoder
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

````{py:attribute} env
:canonical: src.configs.tasks.meta_rl.MetaRLConfig.env
:type: typing.Any
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.meta_rl.MetaRLConfig.env
```

````

`````
