# {py:mod}`src.pipeline.rl.meta.multi_objective`

```{py:module} src.pipeline.rl.meta.multi_objective
```

```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ParetoSolution <src.pipeline.rl.meta.multi_objective.ParetoSolution>`
  - ```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.ParetoSolution
    :summary:
    ```
* - {py:obj}`ParetoFront <src.pipeline.rl.meta.multi_objective.ParetoFront>`
  - ```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.ParetoFront
    :summary:
    ```
* - {py:obj}`MORLWeightOptimizer <src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer>`
  - ```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer
    :summary:
    ```
````

### API

`````{py:class} ParetoSolution(weights, objectives, reward, model_id=None)
:canonical: src.pipeline.rl.meta.multi_objective.ParetoSolution

```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.ParetoSolution
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.ParetoSolution.__init__
```

````{py:method} dominates(other)
:canonical: src.pipeline.rl.meta.multi_objective.ParetoSolution.dominates

```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.ParetoSolution.dominates
```

````

`````

`````{py:class} ParetoFront(max_size=50)
:canonical: src.pipeline.rl.meta.multi_objective.ParetoFront

```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.ParetoFront
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.ParetoFront.__init__
```

````{py:method} add_solution(solution)
:canonical: src.pipeline.rl.meta.multi_objective.ParetoFront.add_solution

```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.ParetoFront.add_solution
```

````

`````

`````{py:class} MORLWeightOptimizer(initial_weights: typing.Dict[str, float], weight_names: typing.List[str] = ['collection', 'cost'], objective_names: typing.List[str] = ['waste_efficiency', 'overflow_rate'], history_window: int = 20, exploration_factor: float = 0.2, adaptation_rate: float = 0.1, **kwargs)
:canonical: src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer

Bases: {py:obj}`logic.src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy`

```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer.__init__
```

````{py:method} propose_weights(context=None)
:canonical: src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer.propose_weights

```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer.propose_weights
```

````

````{py:method} _calculate_objectives(metrics)
:canonical: src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer._calculate_objectives

```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer._calculate_objectives
```

````

````{py:method} update_performance_history(metrics, reward)
:canonical: src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer.update_performance_history

```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer.update_performance_history
```

````

````{py:method} feedback(reward, metrics, day=None, step=None)
:canonical: src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer.feedback

```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer.feedback
```

````

````{py:method} update_weights(metrics=None, reward=None, day=None, step=None)
:canonical: src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer.update_weights

```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer.update_weights
```

````

````{py:method} get_current_weights()
:canonical: src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer.get_current_weights

```{autodoc2-docstring} src.pipeline.rl.meta.multi_objective.MORLWeightOptimizer.get_current_weights
```

````

`````
