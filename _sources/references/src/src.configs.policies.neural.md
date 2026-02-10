# {py:mod}`src.configs.policies.neural`

```{py:module} src.configs.policies.neural
```

```{autodoc2-docstring} src.configs.policies.neural
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuralConfig <src.configs.policies.neural.NeuralConfig>`
  - ```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig
    :summary:
    ```
````

### API

`````{py:class} NeuralConfig
:canonical: src.configs.policies.neural.NeuralConfig

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig
```

````{py:attribute} model
:canonical: src.configs.policies.neural.NeuralConfig.model
:type: src.configs.models.model.ModelConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig.model
```

````

````{py:attribute} decoding
:canonical: src.configs.policies.neural.NeuralConfig.decoding
:type: src.configs.models.decoding.DecodingConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig.decoding
```

````

````{py:attribute} reward
:canonical: src.configs.policies.neural.NeuralConfig.reward
:type: src.configs.envs.objective.ObjectiveConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig.reward
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.neural.NeuralConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.neural.NeuralConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig.post_processing
```

````

`````
