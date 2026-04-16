# {py:mod}`src.policies.route_construction.learning_algorithms.neural_agent.params`

```{py:module} src.policies.route_construction.learning_algorithms.neural_agent.params
```

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuralParams <src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams
    :summary:
    ```
````

### API

`````{py:class} NeuralParams
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams
```

````{py:attribute} waste_weight
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams.waste_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams.waste_weight
```

````

````{py:attribute} cost_weight
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams.cost_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams.cost_weight
```

````

````{py:attribute} overflow_penalty
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams.overflow_penalty
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams.overflow_penalty
```

````

````{py:attribute} selector_name
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams.selector_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams.selector_name
```

````

````{py:attribute} selector_threshold
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams.selector_threshold
:type: float
:value: >
   0.7

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams.selector_threshold
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams.seed
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.params.NeuralParams.to_dict
```

````

`````
