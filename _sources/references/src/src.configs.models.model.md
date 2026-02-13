# {py:mod}`src.configs.models.model`

```{py:module} src.configs.models.model
```

```{autodoc2-docstring} src.configs.models.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ModelConfig <src.configs.models.model.ModelConfig>`
  - ```{autodoc2-docstring} src.configs.models.model.ModelConfig
    :summary:
    ```
````

### API

`````{py:class} ModelConfig
:canonical: src.configs.models.model.ModelConfig

```{autodoc2-docstring} src.configs.models.model.ModelConfig
```

````{py:attribute} name
:canonical: src.configs.models.model.ModelConfig.name
:type: str
:value: >
   'am'

```{autodoc2-docstring} src.configs.models.model.ModelConfig.name
```

````

````{py:attribute} encoder
:canonical: src.configs.models.model.ModelConfig.encoder
:type: src.configs.models.encoder.EncoderConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.models.model.ModelConfig.encoder
```

````

````{py:attribute} decoder
:canonical: src.configs.models.model.ModelConfig.decoder
:type: src.configs.models.decoder.DecoderConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.models.model.ModelConfig.decoder
```

````

````{py:attribute} reward
:canonical: src.configs.models.model.ModelConfig.reward
:type: src.configs.envs.objective.ObjectiveConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.models.model.ModelConfig.reward
```

````

````{py:attribute} temporal_horizon
:canonical: src.configs.models.model.ModelConfig.temporal_horizon
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.models.model.ModelConfig.temporal_horizon
```

````

````{py:attribute} policy_config
:canonical: src.configs.models.model.ModelConfig.policy_config
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.models.model.ModelConfig.policy_config
```

````

````{py:attribute} load_path
:canonical: src.configs.models.model.ModelConfig.load_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.models.model.ModelConfig.load_path
```

````

`````
