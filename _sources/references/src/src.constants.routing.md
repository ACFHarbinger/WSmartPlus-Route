# {py:mod}`src.constants.routing`

```{py:module} src.constants.routing
```

```{autodoc2-docstring} src.constants.routing
:allowtitles:
```

## Module Contents

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IMPROVEMENT_EPSILON <src.constants.routing.IMPROVEMENT_EPSILON>`
  - ```{autodoc2-docstring} src.constants.routing.IMPROVEMENT_EPSILON
    :summary:
    ```
* - {py:obj}`COLLECTION_TIME_MINUTES <src.constants.routing.COLLECTION_TIME_MINUTES>`
  - ```{autodoc2-docstring} src.constants.routing.COLLECTION_TIME_MINUTES
    :summary:
    ```
* - {py:obj}`VEHICLE_SPEED_KMH <src.constants.routing.VEHICLE_SPEED_KMH>`
  - ```{autodoc2-docstring} src.constants.routing.VEHICLE_SPEED_KMH
    :summary:
    ```
* - {py:obj}`PENALTY_MUST_GO_MISSED <src.constants.routing.PENALTY_MUST_GO_MISSED>`
  - ```{autodoc2-docstring} src.constants.routing.PENALTY_MUST_GO_MISSED
    :summary:
    ```
* - {py:obj}`MAX_CAPACITY_PERCENT <src.constants.routing.MAX_CAPACITY_PERCENT>`
  - ```{autodoc2-docstring} src.constants.routing.MAX_CAPACITY_PERCENT
    :summary:
    ```
* - {py:obj}`MIP_GAP <src.constants.routing.MIP_GAP>`
  - ```{autodoc2-docstring} src.constants.routing.MIP_GAP
    :summary:
    ```
* - {py:obj}`HEURISTICS_RATIO <src.constants.routing.HEURISTICS_RATIO>`
  - ```{autodoc2-docstring} src.constants.routing.HEURISTICS_RATIO
    :summary:
    ```
* - {py:obj}`NODEFILE_START_GB <src.constants.routing.NODEFILE_START_GB>`
  - ```{autodoc2-docstring} src.constants.routing.NODEFILE_START_GB
    :summary:
    ```
* - {py:obj}`SOLVER_OUTPUT_FLAG <src.constants.routing.SOLVER_OUTPUT_FLAG>`
  - ```{autodoc2-docstring} src.constants.routing.SOLVER_OUTPUT_FLAG
    :summary:
    ```
* - {py:obj}`DEFAULT_SHIFT_DURATION <src.constants.routing.DEFAULT_SHIFT_DURATION>`
  - ```{autodoc2-docstring} src.constants.routing.DEFAULT_SHIFT_DURATION
    :summary:
    ```
* - {py:obj}`DEFAULT_V_VALUE <src.constants.routing.DEFAULT_V_VALUE>`
  - ```{autodoc2-docstring} src.constants.routing.DEFAULT_V_VALUE
    :summary:
    ```
* - {py:obj}`DEFAULT_COMBINATION <src.constants.routing.DEFAULT_COMBINATION>`
  - ```{autodoc2-docstring} src.constants.routing.DEFAULT_COMBINATION
    :summary:
    ```
* - {py:obj}`DEFAULT_TIME_LIMIT <src.constants.routing.DEFAULT_TIME_LIMIT>`
  - ```{autodoc2-docstring} src.constants.routing.DEFAULT_TIME_LIMIT
    :summary:
    ```
* - {py:obj}`DEFAULT_EVAL_BATCH_SIZE <src.constants.routing.DEFAULT_EVAL_BATCH_SIZE>`
  - ```{autodoc2-docstring} src.constants.routing.DEFAULT_EVAL_BATCH_SIZE
    :summary:
    ```
* - {py:obj}`DEFAULT_ROLLOUT_BATCH_SIZE <src.constants.routing.DEFAULT_ROLLOUT_BATCH_SIZE>`
  - ```{autodoc2-docstring} src.constants.routing.DEFAULT_ROLLOUT_BATCH_SIZE
    :summary:
    ```
````

### API

````{py:data} IMPROVEMENT_EPSILON
:canonical: src.constants.routing.IMPROVEMENT_EPSILON
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.constants.routing.IMPROVEMENT_EPSILON
```

````

````{py:data} COLLECTION_TIME_MINUTES
:canonical: src.constants.routing.COLLECTION_TIME_MINUTES
:value: >
   3.0

```{autodoc2-docstring} src.constants.routing.COLLECTION_TIME_MINUTES
```

````

````{py:data} VEHICLE_SPEED_KMH
:canonical: src.constants.routing.VEHICLE_SPEED_KMH
:value: >
   40.0

```{autodoc2-docstring} src.constants.routing.VEHICLE_SPEED_KMH
```

````

````{py:data} PENALTY_MUST_GO_MISSED
:canonical: src.constants.routing.PENALTY_MUST_GO_MISSED
:value: >
   10000.0

```{autodoc2-docstring} src.constants.routing.PENALTY_MUST_GO_MISSED
```

````

````{py:data} MAX_CAPACITY_PERCENT
:canonical: src.constants.routing.MAX_CAPACITY_PERCENT
:value: >
   100.0

```{autodoc2-docstring} src.constants.routing.MAX_CAPACITY_PERCENT
```

````

````{py:data} MIP_GAP
:canonical: src.constants.routing.MIP_GAP
:value: >
   0.01

```{autodoc2-docstring} src.constants.routing.MIP_GAP
```

````

````{py:data} HEURISTICS_RATIO
:canonical: src.constants.routing.HEURISTICS_RATIO
:value: >
   0.5

```{autodoc2-docstring} src.constants.routing.HEURISTICS_RATIO
```

````

````{py:data} NODEFILE_START_GB
:canonical: src.constants.routing.NODEFILE_START_GB
:value: >
   0.5

```{autodoc2-docstring} src.constants.routing.NODEFILE_START_GB
```

````

````{py:data} SOLVER_OUTPUT_FLAG
:canonical: src.constants.routing.SOLVER_OUTPUT_FLAG
:value: >
   0

```{autodoc2-docstring} src.constants.routing.SOLVER_OUTPUT_FLAG
```

````

````{py:data} DEFAULT_SHIFT_DURATION
:canonical: src.constants.routing.DEFAULT_SHIFT_DURATION
:value: >
   390

```{autodoc2-docstring} src.constants.routing.DEFAULT_SHIFT_DURATION
```

````

````{py:data} DEFAULT_V_VALUE
:canonical: src.constants.routing.DEFAULT_V_VALUE
:value: >
   1.0

```{autodoc2-docstring} src.constants.routing.DEFAULT_V_VALUE
```

````

````{py:data} DEFAULT_COMBINATION
:canonical: src.constants.routing.DEFAULT_COMBINATION
:value: >
   [500, 75, 0.95, 0, 0.095, 0, 0]

```{autodoc2-docstring} src.constants.routing.DEFAULT_COMBINATION
```

````

````{py:data} DEFAULT_TIME_LIMIT
:canonical: src.constants.routing.DEFAULT_TIME_LIMIT
:value: >
   600

```{autodoc2-docstring} src.constants.routing.DEFAULT_TIME_LIMIT
```

````

````{py:data} DEFAULT_EVAL_BATCH_SIZE
:canonical: src.constants.routing.DEFAULT_EVAL_BATCH_SIZE
:type: int
:value: >
   1024

```{autodoc2-docstring} src.constants.routing.DEFAULT_EVAL_BATCH_SIZE
```

````

````{py:data} DEFAULT_ROLLOUT_BATCH_SIZE
:canonical: src.constants.routing.DEFAULT_ROLLOUT_BATCH_SIZE
:type: int
:value: >
   64

```{autodoc2-docstring} src.constants.routing.DEFAULT_ROLLOUT_BATCH_SIZE
```

````
