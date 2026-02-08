# {py:mod}`src.policies.selection.base.selection_context`

```{py:module} src.policies.selection.base.selection_context
```

```{autodoc2-docstring} src.policies.selection.base.selection_context
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SelectionContext <src.policies.selection.base.selection_context.SelectionContext>`
  - ```{autodoc2-docstring} src.policies.selection.base.selection_context.SelectionContext
    :summary:
    ```
````

### API

`````{py:class} SelectionContext
:canonical: src.policies.selection.base.selection_context.SelectionContext

```{autodoc2-docstring} src.policies.selection.base.selection_context.SelectionContext
```

````{py:attribute} bin_ids
:canonical: src.policies.selection.base.selection_context.SelectionContext.bin_ids
:type: numpy.typing.NDArray[numpy.int32]
:value: >
   None

```{autodoc2-docstring} src.policies.selection.base.selection_context.SelectionContext.bin_ids
```

````

````{py:attribute} current_fill
:canonical: src.policies.selection.base.selection_context.SelectionContext.current_fill
:type: numpy.typing.NDArray[numpy.float64]
:value: >
   None

```{autodoc2-docstring} src.policies.selection.base.selection_context.SelectionContext.current_fill
```

````

````{py:attribute} accumulation_rates
:canonical: src.policies.selection.base.selection_context.SelectionContext.accumulation_rates
:type: typing.Optional[numpy.typing.NDArray[numpy.float64]]
:value: >
   None

```{autodoc2-docstring} src.policies.selection.base.selection_context.SelectionContext.accumulation_rates
```

````

````{py:attribute} std_deviations
:canonical: src.policies.selection.base.selection_context.SelectionContext.std_deviations
:type: typing.Optional[numpy.typing.NDArray[numpy.float64]]
:value: >
   None

```{autodoc2-docstring} src.policies.selection.base.selection_context.SelectionContext.std_deviations
```

````

````{py:attribute} current_day
:canonical: src.policies.selection.base.selection_context.SelectionContext.current_day
:type: int
:value: >
   0

```{autodoc2-docstring} src.policies.selection.base.selection_context.SelectionContext.current_day
```

````

````{py:attribute} threshold
:canonical: src.policies.selection.base.selection_context.SelectionContext.threshold
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.selection.base.selection_context.SelectionContext.threshold
```

````

````{py:attribute} next_collection_day
:canonical: src.policies.selection.base.selection_context.SelectionContext.next_collection_day
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.selection.base.selection_context.SelectionContext.next_collection_day
```

````

````{py:attribute} distance_matrix
:canonical: src.policies.selection.base.selection_context.SelectionContext.distance_matrix
:type: typing.Optional[numpy.typing.NDArray[typing.Any]]
:value: >
   None

```{autodoc2-docstring} src.policies.selection.base.selection_context.SelectionContext.distance_matrix
```

````

````{py:attribute} paths_between_states
:canonical: src.policies.selection.base.selection_context.SelectionContext.paths_between_states
:type: typing.Optional[typing.List[typing.List[typing.List[int]]]]
:value: >
   None

```{autodoc2-docstring} src.policies.selection.base.selection_context.SelectionContext.paths_between_states
```

````

````{py:attribute} vehicle_capacity
:canonical: src.policies.selection.base.selection_context.SelectionContext.vehicle_capacity
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.selection.base.selection_context.SelectionContext.vehicle_capacity
```

````

````{py:attribute} revenue_kg
:canonical: src.policies.selection.base.selection_context.SelectionContext.revenue_kg
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.selection.base.selection_context.SelectionContext.revenue_kg
```

````

````{py:attribute} bin_density
:canonical: src.policies.selection.base.selection_context.SelectionContext.bin_density
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.selection.base.selection_context.SelectionContext.bin_density
```

````

````{py:attribute} bin_volume
:canonical: src.policies.selection.base.selection_context.SelectionContext.bin_volume
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.selection.base.selection_context.SelectionContext.bin_volume
```

````

`````
