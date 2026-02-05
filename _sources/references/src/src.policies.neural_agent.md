# {py:mod}`src.policies.neural_agent`

```{py:module} src.policies.neural_agent
```

```{autodoc2-docstring} src.policies.neural_agent
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuralAgent <src.policies.neural_agent.NeuralAgent>`
  - ```{autodoc2-docstring} src.policies.neural_agent.NeuralAgent
    :summary:
    ```
````

### API

`````{py:class} NeuralAgent(model)
:canonical: src.policies.neural_agent.NeuralAgent

```{autodoc2-docstring} src.policies.neural_agent.NeuralAgent
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.neural_agent.NeuralAgent.__init__
```

````{py:method} compute_batch_sim(input, dist_matrix, hrl_manager=None, waste_history=None, threshold=0.5, mask_threshold=0.5)
:canonical: src.policies.neural_agent.NeuralAgent.compute_batch_sim

```{autodoc2-docstring} src.policies.neural_agent.NeuralAgent.compute_batch_sim
```

````

````{py:method} compute_simulator_day(input, graph, distC, profit_vars=None, run_tsp=False, hrl_manager=None, waste_history=None, threshold=0.5, mask_threshold=0.5, two_opt_max_iter=0, cost_weights=None, must_go=None)
:canonical: src.policies.neural_agent.NeuralAgent.compute_simulator_day

```{autodoc2-docstring} src.policies.neural_agent.NeuralAgent.compute_simulator_day
```

````

`````
