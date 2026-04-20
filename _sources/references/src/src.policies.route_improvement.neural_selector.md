# {py:mod}`src.policies.route_improvement.neural_selector`

```{py:module} src.policies.route_improvement.neural_selector
```

```{autodoc2-docstring} src.policies.route_improvement.neural_selector
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OperatorSelectionPolicy <src.policies.route_improvement.neural_selector.OperatorSelectionPolicy>`
  - ```{autodoc2-docstring} src.policies.route_improvement.neural_selector.OperatorSelectionPolicy
    :summary:
    ```
* - {py:obj}`NeuralSelectorRouteImprover <src.policies.route_improvement.neural_selector.NeuralSelectorRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.neural_selector.NeuralSelectorRouteImprover
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_improvement.neural_selector.logger>`
  - ```{autodoc2-docstring} src.policies.route_improvement.neural_selector.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_improvement.neural_selector.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_improvement.neural_selector.logger
```

````

`````{py:class} OperatorSelectionPolicy(state_dim: int, n_operators: int)
:canonical: src.policies.route_improvement.neural_selector.OperatorSelectionPolicy

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.policies.route_improvement.neural_selector.OperatorSelectionPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.neural_selector.OperatorSelectionPolicy.__init__
```

````{py:method} forward(state: torch.Tensor) -> torch.Tensor
:canonical: src.policies.route_improvement.neural_selector.OperatorSelectionPolicy.forward

```{autodoc2-docstring} src.policies.route_improvement.neural_selector.OperatorSelectionPolicy.forward
```

````

`````

`````{py:class} NeuralSelectorRouteImprover(config: typing.Optional[typing.Dict[str, typing.Any]] = None)
:canonical: src.policies.route_improvement.neural_selector.NeuralSelectorRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.neural_selector.NeuralSelectorRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.neural_selector.NeuralSelectorRouteImprover.__init__
```

````{py:method} _extract_state(current_cost: float, best_cost: float, iteration: int, max_iter: int, stagnation: int) -> torch.Tensor
:canonical: src.policies.route_improvement.neural_selector.NeuralSelectorRouteImprover._extract_state

```{autodoc2-docstring} src.policies.route_improvement.neural_selector.NeuralSelectorRouteImprover._extract_state
```

````

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.neural_selector.NeuralSelectorRouteImprover.process

````

`````
