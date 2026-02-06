# {py:mod}`src.policies.selection.vectorized`

```{py:module} src.policies.selection.vectorized
```

```{autodoc2-docstring} src.policies.selection.vectorized
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedSelector <src.policies.selection.vectorized.VectorizedSelector>`
  - ```{autodoc2-docstring} src.policies.selection.vectorized.VectorizedSelector
    :summary:
    ```
* - {py:obj}`LastMinuteSelector <src.policies.selection.vectorized.LastMinuteSelector>`
  - ```{autodoc2-docstring} src.policies.selection.vectorized.LastMinuteSelector
    :summary:
    ```
* - {py:obj}`RegularSelector <src.policies.selection.vectorized.RegularSelector>`
  - ```{autodoc2-docstring} src.policies.selection.vectorized.RegularSelector
    :summary:
    ```
* - {py:obj}`LookaheadSelector <src.policies.selection.vectorized.LookaheadSelector>`
  - ```{autodoc2-docstring} src.policies.selection.vectorized.LookaheadSelector
    :summary:
    ```
* - {py:obj}`RevenueSelector <src.policies.selection.vectorized.RevenueSelector>`
  - ```{autodoc2-docstring} src.policies.selection.vectorized.RevenueSelector
    :summary:
    ```
* - {py:obj}`ServiceLevelSelector <src.policies.selection.vectorized.ServiceLevelSelector>`
  - ```{autodoc2-docstring} src.policies.selection.vectorized.ServiceLevelSelector
    :summary:
    ```
* - {py:obj}`CombinedSelector <src.policies.selection.vectorized.CombinedSelector>`
  - ```{autodoc2-docstring} src.policies.selection.vectorized.CombinedSelector
    :summary:
    ```
* - {py:obj}`ManagerSelector <src.policies.selection.vectorized.ManagerSelector>`
  - ```{autodoc2-docstring} src.policies.selection.vectorized.ManagerSelector
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_selector_from_config <src.policies.selection.vectorized.create_selector_from_config>`
  - ```{autodoc2-docstring} src.policies.selection.vectorized.create_selector_from_config
    :summary:
    ```
* - {py:obj}`get_vectorized_selector <src.policies.selection.vectorized.get_vectorized_selector>`
  - ```{autodoc2-docstring} src.policies.selection.vectorized.get_vectorized_selector
    :summary:
    ```
````

### API

`````{py:class} VectorizedSelector
:canonical: src.policies.selection.vectorized.VectorizedSelector

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.selection.vectorized.VectorizedSelector
```

````{py:method} select(fill_levels: torch.Tensor, **kwargs) -> torch.Tensor
:canonical: src.policies.selection.vectorized.VectorizedSelector.select
:abstractmethod:

```{autodoc2-docstring} src.policies.selection.vectorized.VectorizedSelector.select
```

````

`````

`````{py:class} LastMinuteSelector(threshold: float = 0.7)
:canonical: src.policies.selection.vectorized.LastMinuteSelector

Bases: {py:obj}`src.policies.selection.vectorized.VectorizedSelector`

```{autodoc2-docstring} src.policies.selection.vectorized.LastMinuteSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.selection.vectorized.LastMinuteSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, threshold: typing.Optional[float] = None, **kwargs) -> torch.Tensor
:canonical: src.policies.selection.vectorized.LastMinuteSelector.select

```{autodoc2-docstring} src.policies.selection.vectorized.LastMinuteSelector.select
```

````

`````

`````{py:class} RegularSelector(frequency: int = 3)
:canonical: src.policies.selection.vectorized.RegularSelector

Bases: {py:obj}`src.policies.selection.vectorized.VectorizedSelector`

```{autodoc2-docstring} src.policies.selection.vectorized.RegularSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.selection.vectorized.RegularSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, current_day: typing.Optional[torch.Tensor] = None, frequency: typing.Optional[int] = None, **kwargs) -> torch.Tensor
:canonical: src.policies.selection.vectorized.RegularSelector.select

```{autodoc2-docstring} src.policies.selection.vectorized.RegularSelector.select
```

````

`````

`````{py:class} LookaheadSelector(lookahead_days: int = 1, max_fill: float = 1.0)
:canonical: src.policies.selection.vectorized.LookaheadSelector

Bases: {py:obj}`src.policies.selection.vectorized.VectorizedSelector`

```{autodoc2-docstring} src.policies.selection.vectorized.LookaheadSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.selection.vectorized.LookaheadSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, accumulation_rates: typing.Optional[torch.Tensor] = None, lookahead_days: typing.Optional[int] = None, max_fill: typing.Optional[float] = None, **kwargs) -> torch.Tensor
:canonical: src.policies.selection.vectorized.LookaheadSelector.select

```{autodoc2-docstring} src.policies.selection.vectorized.LookaheadSelector.select
```

````

`````

`````{py:class} RevenueSelector(revenue_kg: float = 1.0, bin_capacity: float = 1.0, threshold: float = 0.0)
:canonical: src.policies.selection.vectorized.RevenueSelector

Bases: {py:obj}`src.policies.selection.vectorized.VectorizedSelector`

```{autodoc2-docstring} src.policies.selection.vectorized.RevenueSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.selection.vectorized.RevenueSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, revenue_kg: typing.Optional[float] = None, bin_capacity: typing.Optional[float] = None, threshold: typing.Optional[float] = None, **kwargs) -> torch.Tensor
:canonical: src.policies.selection.vectorized.RevenueSelector.select

```{autodoc2-docstring} src.policies.selection.vectorized.RevenueSelector.select
```

````

`````

`````{py:class} ServiceLevelSelector(confidence_factor: float = 1.0, max_fill: float = 1.0)
:canonical: src.policies.selection.vectorized.ServiceLevelSelector

Bases: {py:obj}`src.policies.selection.vectorized.VectorizedSelector`

```{autodoc2-docstring} src.policies.selection.vectorized.ServiceLevelSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.selection.vectorized.ServiceLevelSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, accumulation_rates: typing.Optional[torch.Tensor] = None, std_deviations: typing.Optional[torch.Tensor] = None, confidence_factor: typing.Optional[float] = None, max_fill: typing.Optional[float] = None, **kwargs) -> torch.Tensor
:canonical: src.policies.selection.vectorized.ServiceLevelSelector.select

```{autodoc2-docstring} src.policies.selection.vectorized.ServiceLevelSelector.select
```

````

`````

`````{py:class} CombinedSelector(selectors: list[src.policies.selection.vectorized.VectorizedSelector])
:canonical: src.policies.selection.vectorized.CombinedSelector

Bases: {py:obj}`src.policies.selection.vectorized.VectorizedSelector`

```{autodoc2-docstring} src.policies.selection.vectorized.CombinedSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.selection.vectorized.CombinedSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, **kwargs) -> torch.Tensor
:canonical: src.policies.selection.vectorized.CombinedSelector.select

```{autodoc2-docstring} src.policies.selection.vectorized.CombinedSelector.select
```

````

`````

`````{py:class} ManagerSelector(manager=None, manager_config: typing.Optional[dict] = None, threshold: float = 0.5, device: str = 'cuda')
:canonical: src.policies.selection.vectorized.ManagerSelector

Bases: {py:obj}`src.policies.selection.vectorized.VectorizedSelector`

```{autodoc2-docstring} src.policies.selection.vectorized.ManagerSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.selection.vectorized.ManagerSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, locs: typing.Optional[torch.Tensor] = None, waste_history: typing.Optional[torch.Tensor] = None, threshold: typing.Optional[float] = None, **kwargs) -> torch.Tensor
:canonical: src.policies.selection.vectorized.ManagerSelector.select

```{autodoc2-docstring} src.policies.selection.vectorized.ManagerSelector.select
```

````

````{py:method} load_weights(path: str)
:canonical: src.policies.selection.vectorized.ManagerSelector.load_weights

```{autodoc2-docstring} src.policies.selection.vectorized.ManagerSelector.load_weights
```

````

`````

````{py:function} create_selector_from_config(cfg) -> typing.Optional[src.policies.selection.vectorized.VectorizedSelector]
:canonical: src.policies.selection.vectorized.create_selector_from_config

```{autodoc2-docstring} src.policies.selection.vectorized.create_selector_from_config
```
````

````{py:function} get_vectorized_selector(name: str, **kwargs) -> src.policies.selection.vectorized.VectorizedSelector
:canonical: src.policies.selection.vectorized.get_vectorized_selector

```{autodoc2-docstring} src.policies.selection.vectorized.get_vectorized_selector
```
````
