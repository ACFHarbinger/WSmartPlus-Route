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

````{py:attribute} model_path
:canonical: src.configs.policies.neural.NeuralConfig.model_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig.model_path
```

````

````{py:attribute} decode_type
:canonical: src.configs.policies.neural.NeuralConfig.decode_type
:type: str
:value: >
   'greedy'

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig.decode_type
```

````

````{py:attribute} temperature
:canonical: src.configs.policies.neural.NeuralConfig.temperature
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig.temperature
```

````

````{py:attribute} beam_width
:canonical: src.configs.policies.neural.NeuralConfig.beam_width
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig.beam_width
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.neural.NeuralConfig.must_go
:type: typing.Optional[typing.List[str]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.neural.NeuralConfig.post_processing
:type: typing.Optional[typing.List[str]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig.post_processing
```

````

`````
