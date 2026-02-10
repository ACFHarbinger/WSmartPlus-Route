# {py:mod}`src.policies.neural_agent.simulation`

```{py:module} src.policies.neural_agent.simulation
```

```{autodoc2-docstring} src.policies.neural_agent.simulation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationMixin <src.policies.neural_agent.simulation.SimulationMixin>`
  - ```{autodoc2-docstring} src.policies.neural_agent.simulation.SimulationMixin
    :summary:
    ```
````

### API

`````{py:class} SimulationMixin
:canonical: src.policies.neural_agent.simulation.SimulationMixin

```{autodoc2-docstring} src.policies.neural_agent.simulation.SimulationMixin
```

````{py:method} compute_simulator_day(input, graph, distC, profit_vars=None, waste_history=None, cost_weights=None, must_go=None)
:canonical: src.policies.neural_agent.simulation.SimulationMixin.compute_simulator_day

```{autodoc2-docstring} src.policies.neural_agent.simulation.SimulationMixin.compute_simulator_day
```

````

````{py:method} _validate_must_go(must_go)
:canonical: src.policies.neural_agent.simulation.SimulationMixin._validate_must_go

```{autodoc2-docstring} src.policies.neural_agent.simulation.SimulationMixin._validate_must_go
```

````

````{py:method} _prepare_temporal_features(input_for_model, waste_history)
:canonical: src.policies.neural_agent.simulation.SimulationMixin._prepare_temporal_features

```{autodoc2-docstring} src.policies.neural_agent.simulation.SimulationMixin._prepare_temporal_features
```

````

`````
