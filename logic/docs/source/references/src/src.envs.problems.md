# {py:mod}`src.envs.problems`

```{py:module} src.envs.problems
```

```{autodoc2-docstring} src.envs.problems
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BaseProblem <src.envs.problems.BaseProblem>`
  - ```{autodoc2-docstring} src.envs.problems.BaseProblem
    :summary:
    ```
* - {py:obj}`VRPP <src.envs.problems.VRPP>`
  - ```{autodoc2-docstring} src.envs.problems.VRPP
    :summary:
    ```
* - {py:obj}`CVRPP <src.envs.problems.CVRPP>`
  - ```{autodoc2-docstring} src.envs.problems.CVRPP
    :summary:
    ```
* - {py:obj}`WCVRP <src.envs.problems.WCVRP>`
  - ```{autodoc2-docstring} src.envs.problems.WCVRP
    :summary:
    ```
* - {py:obj}`CWCVRP <src.envs.problems.CWCVRP>`
  - ```{autodoc2-docstring} src.envs.problems.CWCVRP
    :summary:
    ```
* - {py:obj}`SDWCVRP <src.envs.problems.SDWCVRP>`
  - ```{autodoc2-docstring} src.envs.problems.SDWCVRP
    :summary:
    ```
* - {py:obj}`SCWCVRP <src.envs.problems.SCWCVRP>`
  - ```{autodoc2-docstring} src.envs.problems.SCWCVRP
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`COST_KM <src.envs.problems.COST_KM>`
  - ```{autodoc2-docstring} src.envs.problems.COST_KM
    :summary:
    ```
* - {py:obj}`REVENUE_KG <src.envs.problems.REVENUE_KG>`
  - ```{autodoc2-docstring} src.envs.problems.REVENUE_KG
    :summary:
    ```
* - {py:obj}`BIN_CAPACITY <src.envs.problems.BIN_CAPACITY>`
  - ```{autodoc2-docstring} src.envs.problems.BIN_CAPACITY
    :summary:
    ```
* - {py:obj}`VEHICLE_CAPACITY <src.envs.problems.VEHICLE_CAPACITY>`
  - ```{autodoc2-docstring} src.envs.problems.VEHICLE_CAPACITY
    :summary:
    ```
````

### API

````{py:data} COST_KM
:canonical: src.envs.problems.COST_KM
:value: >
   1.0

```{autodoc2-docstring} src.envs.problems.COST_KM
```

````

````{py:data} REVENUE_KG
:canonical: src.envs.problems.REVENUE_KG
:value: >
   1.0

```{autodoc2-docstring} src.envs.problems.REVENUE_KG
```

````

````{py:data} BIN_CAPACITY
:canonical: src.envs.problems.BIN_CAPACITY
:value: >
   100.0

```{autodoc2-docstring} src.envs.problems.BIN_CAPACITY
```

````

````{py:data} VEHICLE_CAPACITY
:canonical: src.envs.problems.VEHICLE_CAPACITY
:value: >
   100.0

```{autodoc2-docstring} src.envs.problems.VEHICLE_CAPACITY
```

````

`````{py:class} BaseProblem
:canonical: src.envs.problems.BaseProblem

```{autodoc2-docstring} src.envs.problems.BaseProblem
```

````{py:attribute} NAME
:canonical: src.envs.problems.BaseProblem.NAME
:type: str
:value: >
   'base'

```{autodoc2-docstring} src.envs.problems.BaseProblem.NAME
```

````

````{py:method} validate_tours(pi: torch.Tensor) -> bool
:canonical: src.envs.problems.BaseProblem.validate_tours
:staticmethod:

```{autodoc2-docstring} src.envs.problems.BaseProblem.validate_tours
```

````

````{py:method} get_tour_length(dataset: typing.Dict[str, typing.Any], pi: torch.Tensor, dist_matrix: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.problems.BaseProblem.get_tour_length
:staticmethod:

```{autodoc2-docstring} src.envs.problems.BaseProblem.get_tour_length
```

````

````{py:method} beam_search(input, beam_size, cost_weights, model=None, **kwargs)
:canonical: src.envs.problems.BaseProblem.beam_search
:classmethod:

```{autodoc2-docstring} src.envs.problems.BaseProblem.beam_search
```

````

````{py:method} make_state(input_data: typing.Any, edges: typing.Any = None, cost_weights: typing.Any = None, dist_matrix: typing.Any = None, **kwargs: typing.Any) -> typing.Any
:canonical: src.envs.problems.BaseProblem.make_state
:classmethod:

```{autodoc2-docstring} src.envs.problems.BaseProblem.make_state
```

````

`````

`````{py:class} VRPP
:canonical: src.envs.problems.VRPP

Bases: {py:obj}`src.envs.problems.BaseProblem`

```{autodoc2-docstring} src.envs.problems.VRPP
```

````{py:attribute} NAME
:canonical: src.envs.problems.VRPP.NAME
:value: >
   'vrpp'

```{autodoc2-docstring} src.envs.problems.VRPP.NAME
```

````

````{py:method} get_costs(dataset, pi, cw_dict, dist_matrix=None)
:canonical: src.envs.problems.VRPP.get_costs
:staticmethod:

```{autodoc2-docstring} src.envs.problems.VRPP.get_costs
```

````

`````

`````{py:class} CVRPP
:canonical: src.envs.problems.CVRPP

Bases: {py:obj}`src.envs.problems.VRPP`

```{autodoc2-docstring} src.envs.problems.CVRPP
```

````{py:attribute} NAME
:canonical: src.envs.problems.CVRPP.NAME
:value: >
   'cvrpp'

```{autodoc2-docstring} src.envs.problems.CVRPP.NAME
```

````

````{py:method} get_costs(dataset, pi, cw_dict, dist_matrix=None)
:canonical: src.envs.problems.CVRPP.get_costs
:staticmethod:

```{autodoc2-docstring} src.envs.problems.CVRPP.get_costs
```

````

`````

`````{py:class} WCVRP
:canonical: src.envs.problems.WCVRP

Bases: {py:obj}`src.envs.problems.BaseProblem`

```{autodoc2-docstring} src.envs.problems.WCVRP
```

````{py:attribute} NAME
:canonical: src.envs.problems.WCVRP.NAME
:value: >
   'wcvrp'

```{autodoc2-docstring} src.envs.problems.WCVRP.NAME
```

````

````{py:method} get_costs(dataset, pi, cw_dict, dist_matrix=None)
:canonical: src.envs.problems.WCVRP.get_costs
:staticmethod:

```{autodoc2-docstring} src.envs.problems.WCVRP.get_costs
```

````

`````

`````{py:class} CWCVRP
:canonical: src.envs.problems.CWCVRP

Bases: {py:obj}`src.envs.problems.WCVRP`

```{autodoc2-docstring} src.envs.problems.CWCVRP
```

````{py:attribute} NAME
:canonical: src.envs.problems.CWCVRP.NAME
:value: >
   'cwcvrp'

```{autodoc2-docstring} src.envs.problems.CWCVRP.NAME
```

````

`````

`````{py:class} SDWCVRP
:canonical: src.envs.problems.SDWCVRP

Bases: {py:obj}`src.envs.problems.WCVRP`

```{autodoc2-docstring} src.envs.problems.SDWCVRP
```

````{py:attribute} NAME
:canonical: src.envs.problems.SDWCVRP.NAME
:value: >
   'sdwcvrp'

```{autodoc2-docstring} src.envs.problems.SDWCVRP.NAME
```

````

`````

`````{py:class} SCWCVRP
:canonical: src.envs.problems.SCWCVRP

Bases: {py:obj}`src.envs.problems.WCVRP`

```{autodoc2-docstring} src.envs.problems.SCWCVRP
```

````{py:attribute} NAME
:canonical: src.envs.problems.SCWCVRP.NAME
:value: >
   'scwcvrp'

```{autodoc2-docstring} src.envs.problems.SCWCVRP.NAME
```

````

`````
