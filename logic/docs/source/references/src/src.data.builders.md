# {py:mod}`src.data.builders`

```{py:module} src.data.builders
```

```{autodoc2-docstring} src.data.builders
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VRPInstanceBuilder <src.data.builders.VRPInstanceBuilder>`
  - ```{autodoc2-docstring} src.data.builders.VRPInstanceBuilder
    :summary:
    ```
````

### API

`````{py:class} VRPInstanceBuilder(data=None, depot_idx=0, vehicle_cap=100.0, customers=None, dimension=0, coords=None)
:canonical: src.data.builders.VRPInstanceBuilder

```{autodoc2-docstring} src.data.builders.VRPInstanceBuilder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.builders.VRPInstanceBuilder.__init__
```

````{py:method} set_dataset_size(size: int)
:canonical: src.data.builders.VRPInstanceBuilder.set_dataset_size

```{autodoc2-docstring} src.data.builders.VRPInstanceBuilder.set_dataset_size
```

````

````{py:method} set_problem_size(size: int)
:canonical: src.data.builders.VRPInstanceBuilder.set_problem_size

```{autodoc2-docstring} src.data.builders.VRPInstanceBuilder.set_problem_size
```

````

````{py:method} set_waste_type(waste_type: str)
:canonical: src.data.builders.VRPInstanceBuilder.set_waste_type

```{autodoc2-docstring} src.data.builders.VRPInstanceBuilder.set_waste_type
```

````

````{py:method} set_distribution(distribution: str)
:canonical: src.data.builders.VRPInstanceBuilder.set_distribution

```{autodoc2-docstring} src.data.builders.VRPInstanceBuilder.set_distribution
```

````

````{py:method} set_area(area: str)
:canonical: src.data.builders.VRPInstanceBuilder.set_area

```{autodoc2-docstring} src.data.builders.VRPInstanceBuilder.set_area
```

````

````{py:method} set_focus_graph(focus_graph: typing.Optional[str] = None, focus_size: int = 0)
:canonical: src.data.builders.VRPInstanceBuilder.set_focus_graph

```{autodoc2-docstring} src.data.builders.VRPInstanceBuilder.set_focus_graph
```

````

````{py:method} set_method(method: str)
:canonical: src.data.builders.VRPInstanceBuilder.set_method

```{autodoc2-docstring} src.data.builders.VRPInstanceBuilder.set_method
```

````

````{py:method} set_num_days(num_days: int)
:canonical: src.data.builders.VRPInstanceBuilder.set_num_days

```{autodoc2-docstring} src.data.builders.VRPInstanceBuilder.set_num_days
```

````

````{py:method} set_problem_name(problem_name: str)
:canonical: src.data.builders.VRPInstanceBuilder.set_problem_name

```{autodoc2-docstring} src.data.builders.VRPInstanceBuilder.set_problem_name
```

````

````{py:method} set_noise(mean: float, variance: float)
:canonical: src.data.builders.VRPInstanceBuilder.set_noise

```{autodoc2-docstring} src.data.builders.VRPInstanceBuilder.set_noise
```

````

````{py:method} build()
:canonical: src.data.builders.VRPInstanceBuilder.build

```{autodoc2-docstring} src.data.builders.VRPInstanceBuilder.build
```

````

````{py:method} build_td() -> tensordict.TensorDict
:canonical: src.data.builders.VRPInstanceBuilder.build_td

```{autodoc2-docstring} src.data.builders.VRPInstanceBuilder.build_td
```

````

````{py:method} _prepare_coordinates()
:canonical: src.data.builders.VRPInstanceBuilder._prepare_coordinates

```{autodoc2-docstring} src.data.builders.VRPInstanceBuilder._prepare_coordinates
```

````

`````
