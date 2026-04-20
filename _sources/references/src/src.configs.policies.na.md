# {py:mod}`src.configs.policies.na`

```{py:module} src.configs.policies.na
```

```{autodoc2-docstring} src.configs.policies.na
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuralAgentConfig <src.configs.policies.na.NeuralAgentConfig>`
  - ```{autodoc2-docstring} src.configs.policies.na.NeuralAgentConfig
    :summary:
    ```
````

### API

`````{py:class} NeuralAgentConfig
:canonical: src.configs.policies.na.NeuralAgentConfig

```{autodoc2-docstring} src.configs.policies.na.NeuralAgentConfig
```

````{py:attribute} model
:canonical: src.configs.policies.na.NeuralAgentConfig.model
:type: logic.src.configs.models.model.ModelConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.na.NeuralAgentConfig.model
```

````

````{py:attribute} reward
:canonical: src.configs.policies.na.NeuralAgentConfig.reward
:type: logic.src.configs.envs.objective.ObjectiveConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.na.NeuralAgentConfig.reward
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.na.NeuralAgentConfig.mandatory_selection
:type: typing.Optional[typing.List[logic.src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.na.NeuralAgentConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.na.NeuralAgentConfig.route_improvement
:type: typing.Optional[typing.List[logic.src.configs.policies.other.route_improvement.RouteImprovingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.na.NeuralAgentConfig.route_improvement
```

````

````{py:attribute} seed
:canonical: src.configs.policies.na.NeuralAgentConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.na.NeuralAgentConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.na.NeuralAgentConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.na.NeuralAgentConfig.vrpp
```

````

`````
