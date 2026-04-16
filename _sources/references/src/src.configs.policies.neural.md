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

````{py:attribute} reward
:canonical: src.configs.policies.neural.NeuralConfig.reward
:type: src.configs.envs.objective.ObjectiveConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig.reward
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.neural.NeuralConfig.mandatory_selection
:type: typing.Optional[typing.List[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.neural.NeuralConfig.route_improvement
:type: typing.Optional[typing.List[src.configs.policies.other.route_improvement.RouteImprovingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig.route_improvement
```

````

````{py:attribute} seed
:canonical: src.configs.policies.neural.NeuralConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.neural.NeuralConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.neural.NeuralConfig.vrpp
```

````

`````
