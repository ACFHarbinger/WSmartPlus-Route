# {py:mod}`src.constants.simulation`

```{py:module} src.constants.simulation
```

```{autodoc2-docstring} src.constants.simulation
:allowtitles:
```

## Module Contents

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EARTH_RADIUS <src.constants.simulation.EARTH_RADIUS>`
  - ```{autodoc2-docstring} src.constants.simulation.EARTH_RADIUS
    :summary:
    ```
* - {py:obj}`EARTH_WMP_RADIUS <src.constants.simulation.EARTH_WMP_RADIUS>`
  - ```{autodoc2-docstring} src.constants.simulation.EARTH_WMP_RADIUS
    :summary:
    ```
* - {py:obj}`METRICS <src.constants.simulation.METRICS>`
  - ```{autodoc2-docstring} src.constants.simulation.METRICS
    :summary:
    ```
* - {py:obj}`SIM_METRICS <src.constants.simulation.SIM_METRICS>`
  - ```{autodoc2-docstring} src.constants.simulation.SIM_METRICS
    :summary:
    ```
* - {py:obj}`DAY_METRICS <src.constants.simulation.DAY_METRICS>`
  - ```{autodoc2-docstring} src.constants.simulation.DAY_METRICS
    :summary:
    ```
* - {py:obj}`LOSS_KEYS <src.constants.simulation.LOSS_KEYS>`
  - ```{autodoc2-docstring} src.constants.simulation.LOSS_KEYS
    :summary:
    ```
* - {py:obj}`MAX_WASTE <src.constants.simulation.MAX_WASTE>`
  - ```{autodoc2-docstring} src.constants.simulation.MAX_WASTE
    :summary:
    ```
* - {py:obj}`MAX_LENGTHS <src.constants.simulation.MAX_LENGTHS>`
  - ```{autodoc2-docstring} src.constants.simulation.MAX_LENGTHS
    :summary:
    ```
* - {py:obj}`VEHICLE_CAPACITY <src.constants.simulation.VEHICLE_CAPACITY>`
  - ```{autodoc2-docstring} src.constants.simulation.VEHICLE_CAPACITY
    :summary:
    ```
* - {py:obj}`PROBLEMS <src.constants.simulation.PROBLEMS>`
  - ```{autodoc2-docstring} src.constants.simulation.PROBLEMS
    :summary:
    ```
````

### API

````{py:data} EARTH_RADIUS
:canonical: src.constants.simulation.EARTH_RADIUS
:type: int
:value: >
   6371

```{autodoc2-docstring} src.constants.simulation.EARTH_RADIUS
```

````

````{py:data} EARTH_WMP_RADIUS
:canonical: src.constants.simulation.EARTH_WMP_RADIUS
:type: int
:value: >
   6378137

```{autodoc2-docstring} src.constants.simulation.EARTH_WMP_RADIUS
```

````

````{py:data} METRICS
:canonical: src.constants.simulation.METRICS
:type: typing.List[str]
:value: >
   ['overflows', 'kg', 'ncol', 'kg_lost', 'km', 'kg/km', 'cost', 'profit']

```{autodoc2-docstring} src.constants.simulation.METRICS
```

````

````{py:data} SIM_METRICS
:canonical: src.constants.simulation.SIM_METRICS
:type: typing.List[str]
:value: >
   None

```{autodoc2-docstring} src.constants.simulation.SIM_METRICS
```

````

````{py:data} DAY_METRICS
:canonical: src.constants.simulation.DAY_METRICS
:type: typing.List[str]
:value: >
   None

```{autodoc2-docstring} src.constants.simulation.DAY_METRICS
```

````

````{py:data} LOSS_KEYS
:canonical: src.constants.simulation.LOSS_KEYS
:type: typing.List[str]
:value: >
   ['nll', 'reinforce_loss', 'baseline_loss']

```{autodoc2-docstring} src.constants.simulation.LOSS_KEYS
```

````

````{py:data} MAX_WASTE
:canonical: src.constants.simulation.MAX_WASTE
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.constants.simulation.MAX_WASTE
```

````

````{py:data} MAX_LENGTHS
:canonical: src.constants.simulation.MAX_LENGTHS
:type: typing.Dict[int, int]
:value: >
   None

```{autodoc2-docstring} src.constants.simulation.MAX_LENGTHS
```

````

````{py:data} VEHICLE_CAPACITY
:canonical: src.constants.simulation.VEHICLE_CAPACITY
:type: float
:value: >
   200.0

```{autodoc2-docstring} src.constants.simulation.VEHICLE_CAPACITY
```

````

````{py:data} PROBLEMS
:canonical: src.constants.simulation.PROBLEMS
:type: typing.List[str]
:value: >
   ['vrpp', 'cvrpp', 'wcvrp', 'cwcvrp', 'sdwcvrp', 'scwcvrp']

```{autodoc2-docstring} src.constants.simulation.PROBLEMS
```

````
