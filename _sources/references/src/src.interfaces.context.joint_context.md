# {py:mod}`src.interfaces.context.joint_context`

```{py:module} src.interfaces.context.joint_context
```

```{autodoc2-docstring} src.interfaces.context.joint_context
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointSelectionConstructionContext <src.interfaces.context.joint_context.JointSelectionConstructionContext>`
  - ```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext
    :summary:
    ```
````

### API

`````{py:class} JointSelectionConstructionContext
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext
```

````{py:attribute} bin_ids
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.bin_ids
:type: numpy.typing.NDArray[numpy.int32]
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.bin_ids
```

````

````{py:attribute} current_fill
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.current_fill
:type: numpy.typing.NDArray[numpy.float64]
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.current_fill
```

````

````{py:attribute} distance_matrix
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.distance_matrix
:type: numpy.typing.NDArray[typing.Any]
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.distance_matrix
```

````

````{py:attribute} capacity
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.capacity
:type: float
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.capacity
```

````

````{py:attribute} revenue_kg
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.revenue_kg
:type: float
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.revenue_kg
```

````

````{py:attribute} cost_per_km
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.cost_per_km
:type: float
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.cost_per_km
```

````

````{py:attribute} accumulation_rates
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.accumulation_rates
:type: typing.Optional[numpy.typing.NDArray[numpy.float64]]
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.accumulation_rates
```

````

````{py:attribute} std_deviations
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.std_deviations
:type: typing.Optional[numpy.typing.NDArray[numpy.float64]]
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.std_deviations
```

````

````{py:attribute} bin_density
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.bin_density
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.bin_density
```

````

````{py:attribute} bin_volume
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.bin_volume
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.bin_volume
```

````

````{py:attribute} max_fill
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.max_fill
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.max_fill
```

````

````{py:attribute} overflow_penalty_frac
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.overflow_penalty_frac
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.overflow_penalty_frac
```

````

````{py:attribute} n_vehicles
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.n_vehicles
:type: int
:value: >
   1

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.n_vehicles
```

````

````{py:attribute} current_day
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.current_day
:type: int
:value: >
   0

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.current_day
```

````

````{py:attribute} horizon_days
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.horizon_days
:type: int
:value: >
   3

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.horizon_days
```

````

````{py:attribute} scenario_tree
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.scenario_tree
:type: typing.Optional[typing.Any]
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.scenario_tree
```

````

````{py:attribute} paths_between_states
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.paths_between_states
:type: typing.Optional[typing.List[typing.List[typing.List[int]]]]
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.paths_between_states
```

````

````{py:attribute} mandatory_override
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.mandatory_override
:type: typing.Optional[typing.List[int]]
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.mandatory_override
```

````

````{py:method} bin_mass_kg() -> numpy.typing.NDArray[numpy.float64]
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.bin_mass_kg

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.bin_mass_kg
```

````

````{py:method} revenue_scaled() -> float
:canonical: src.interfaces.context.joint_context.JointSelectionConstructionContext.revenue_scaled

```{autodoc2-docstring} src.interfaces.context.joint_context.JointSelectionConstructionContext.revenue_scaled
```

````

`````
