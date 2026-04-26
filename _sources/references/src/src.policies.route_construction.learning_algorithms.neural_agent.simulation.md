# {py:mod}`src.policies.route_construction.learning_algorithms.neural_agent.simulation`

```{py:module} src.policies.route_construction.learning_algorithms.neural_agent.simulation
```

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.simulation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationMixin <src.policies.route_construction.learning_algorithms.neural_agent.simulation.SimulationMixin>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.simulation.SimulationMixin
    :summary:
    ```
````

### API

`````{py:class} SimulationMixin
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.simulation.SimulationMixin

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.simulation.SimulationMixin
```

````{py:method} compute_simulator_day(input: typing.Dict[str, typing.Any], graph: typing.Tuple[typing.Any, typing.Any], distC: torch.Tensor, profit_vars: typing.Optional[typing.Any] = None, waste_history: typing.Optional[torch.Tensor] = None, cost_weights: typing.Optional[typing.Dict[str, float]] = None, mandatory: typing.Optional[torch.Tensor] = None, **kwargs: typing.Any)
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.simulation.SimulationMixin.compute_simulator_day

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.simulation.SimulationMixin.compute_simulator_day
```

````

````{py:method} _validate_mandatory(mandatory: torch.Tensor) -> torch.Tensor
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.simulation.SimulationMixin._validate_mandatory

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.simulation.SimulationMixin._validate_mandatory
```

````

````{py:method} _prepare_temporal_features(input_for_model: typing.Dict[str, typing.Any], waste_history: torch.Tensor)
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.simulation.SimulationMixin._prepare_temporal_features

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.simulation.SimulationMixin._prepare_temporal_features
```

````

`````
