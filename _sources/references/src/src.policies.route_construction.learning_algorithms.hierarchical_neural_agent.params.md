# {py:mod}`src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params`

```{py:module} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params
```

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HNAParams <src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams
    :summary:
    ```
````

### API

`````{py:class} HNAParams
:canonical: src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams
```

````{py:attribute} checkpoint_path
:canonical: src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.checkpoint_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.checkpoint_path
```

````

````{py:attribute} device
:canonical: src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.device
:type: str
:value: >
   'cpu'

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.device
```

````

````{py:attribute} horizon
:canonical: src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.horizon
:type: int
:value: >
   7

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.horizon
```

````

````{py:attribute} overflow_penalty
:canonical: src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.overflow_penalty
:type: float
:value: >
   500.0

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.overflow_penalty
```

````

````{py:attribute} greedy_threshold
:canonical: src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.greedy_threshold
:type: float
:value: >
   75.0

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.greedy_threshold
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.seed
```

````

````{py:attribute} verbose
:canonical: src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.verbose
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.verbose
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams
:canonical: src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params.HNAParams.to_dict
```

````

`````
