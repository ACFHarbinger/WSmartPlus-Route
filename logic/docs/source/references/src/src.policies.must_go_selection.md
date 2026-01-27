# {py:mod}`src.policies.must_go_selection`

```{py:module} src.policies.must_go_selection
```

```{autodoc2-docstring} src.policies.must_go_selection
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SelectionContext <src.policies.must_go_selection.SelectionContext>`
  - ```{autodoc2-docstring} src.policies.must_go_selection.SelectionContext
    :summary:
    ```
* - {py:obj}`MustGoSelectionStrategy <src.policies.must_go_selection.MustGoSelectionStrategy>`
  - ```{autodoc2-docstring} src.policies.must_go_selection.MustGoSelectionStrategy
    :summary:
    ```
* - {py:obj}`MustGoSelectionRegistry <src.policies.must_go_selection.MustGoSelectionRegistry>`
  - ```{autodoc2-docstring} src.policies.must_go_selection.MustGoSelectionRegistry
    :summary:
    ```
* - {py:obj}`MustGoSelectionFactory <src.policies.must_go_selection.MustGoSelectionFactory>`
  - ```{autodoc2-docstring} src.policies.must_go_selection.MustGoSelectionFactory
    :summary:
    ```
````

### API

`````{py:class} SelectionContext
:canonical: src.policies.must_go_selection.SelectionContext

```{autodoc2-docstring} src.policies.must_go_selection.SelectionContext
```

````{py:attribute} bin_ids
:canonical: src.policies.must_go_selection.SelectionContext.bin_ids
:type: numpy.typing.NDArray[numpy.int32]
:value: >
   None

```{autodoc2-docstring} src.policies.must_go_selection.SelectionContext.bin_ids
```

````

````{py:attribute} current_fill
:canonical: src.policies.must_go_selection.SelectionContext.current_fill
:type: numpy.typing.NDArray[numpy.float64]
:value: >
   None

```{autodoc2-docstring} src.policies.must_go_selection.SelectionContext.current_fill
```

````

````{py:attribute} accumulation_rates
:canonical: src.policies.must_go_selection.SelectionContext.accumulation_rates
:type: typing.Optional[numpy.typing.NDArray[numpy.float64]]
:value: >
   None

```{autodoc2-docstring} src.policies.must_go_selection.SelectionContext.accumulation_rates
```

````

````{py:attribute} std_deviations
:canonical: src.policies.must_go_selection.SelectionContext.std_deviations
:type: typing.Optional[numpy.typing.NDArray[numpy.float64]]
:value: >
   None

```{autodoc2-docstring} src.policies.must_go_selection.SelectionContext.std_deviations
```

````

````{py:attribute} current_day
:canonical: src.policies.must_go_selection.SelectionContext.current_day
:type: int
:value: >
   0

```{autodoc2-docstring} src.policies.must_go_selection.SelectionContext.current_day
```

````

````{py:attribute} threshold
:canonical: src.policies.must_go_selection.SelectionContext.threshold
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.must_go_selection.SelectionContext.threshold
```

````

````{py:attribute} next_collection_day
:canonical: src.policies.must_go_selection.SelectionContext.next_collection_day
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.must_go_selection.SelectionContext.next_collection_day
```

````

````{py:attribute} distance_matrix
:canonical: src.policies.must_go_selection.SelectionContext.distance_matrix
:type: typing.Optional[numpy.typing.NDArray[typing.Any]]
:value: >
   None

```{autodoc2-docstring} src.policies.must_go_selection.SelectionContext.distance_matrix
```

````

````{py:attribute} paths_between_states
:canonical: src.policies.must_go_selection.SelectionContext.paths_between_states
:type: typing.Optional[typing.List[typing.List[typing.List[int]]]]
:value: >
   None

```{autodoc2-docstring} src.policies.must_go_selection.SelectionContext.paths_between_states
```

````

````{py:attribute} vehicle_capacity
:canonical: src.policies.must_go_selection.SelectionContext.vehicle_capacity
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.must_go_selection.SelectionContext.vehicle_capacity
```

````

````{py:attribute} revenue_kg
:canonical: src.policies.must_go_selection.SelectionContext.revenue_kg
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.must_go_selection.SelectionContext.revenue_kg
```

````

````{py:attribute} bin_density
:canonical: src.policies.must_go_selection.SelectionContext.bin_density
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.must_go_selection.SelectionContext.bin_density
```

````

````{py:attribute} bin_volume
:canonical: src.policies.must_go_selection.SelectionContext.bin_volume
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.must_go_selection.SelectionContext.bin_volume
```

````

````{py:attribute} lookahead_days
:canonical: src.policies.must_go_selection.SelectionContext.lookahead_days
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.must_go_selection.SelectionContext.lookahead_days
```

````

`````

`````{py:class} MustGoSelectionStrategy
:canonical: src.policies.must_go_selection.MustGoSelectionStrategy

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.must_go_selection.MustGoSelectionStrategy
```

````{py:method} select_bins(context: src.policies.must_go_selection.SelectionContext) -> typing.List[int]
:canonical: src.policies.must_go_selection.MustGoSelectionStrategy.select_bins
:abstractmethod:

```{autodoc2-docstring} src.policies.must_go_selection.MustGoSelectionStrategy.select_bins
```

````

`````

`````{py:class} MustGoSelectionRegistry
:canonical: src.policies.must_go_selection.MustGoSelectionRegistry

```{autodoc2-docstring} src.policies.must_go_selection.MustGoSelectionRegistry
```

````{py:attribute} _strategies
:canonical: src.policies.must_go_selection.MustGoSelectionRegistry._strategies
:type: typing.Dict[str, typing.Type[src.policies.must_go_selection.MustGoSelectionStrategy]]
:value: >
   None

```{autodoc2-docstring} src.policies.must_go_selection.MustGoSelectionRegistry._strategies
```

````

````{py:method} register(name: str)
:canonical: src.policies.must_go_selection.MustGoSelectionRegistry.register
:classmethod:

```{autodoc2-docstring} src.policies.must_go_selection.MustGoSelectionRegistry.register
```

````

````{py:method} get_strategy_class(name: str) -> typing.Optional[typing.Type[src.policies.must_go_selection.MustGoSelectionStrategy]]
:canonical: src.policies.must_go_selection.MustGoSelectionRegistry.get_strategy_class
:classmethod:

```{autodoc2-docstring} src.policies.must_go_selection.MustGoSelectionRegistry.get_strategy_class
```

````

`````

`````{py:class} MustGoSelectionFactory
:canonical: src.policies.must_go_selection.MustGoSelectionFactory

```{autodoc2-docstring} src.policies.must_go_selection.MustGoSelectionFactory
```

````{py:method} create_strategy(name: str) -> src.policies.must_go_selection.MustGoSelectionStrategy
:canonical: src.policies.must_go_selection.MustGoSelectionFactory.create_strategy
:staticmethod:

```{autodoc2-docstring} src.policies.must_go_selection.MustGoSelectionFactory.create_strategy
```

````

`````
