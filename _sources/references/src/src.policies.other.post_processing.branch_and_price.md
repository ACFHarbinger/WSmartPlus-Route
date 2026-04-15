# {py:mod}`src.policies.other.post_processing.branch_and_price`

```{py:module} src.policies.other.post_processing.branch_and_price
```

```{autodoc2-docstring} src.policies.other.post_processing.branch_and_price
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BranchAndPricePostProcessor <src.policies.other.post_processing.branch_and_price.BranchAndPricePostProcessor>`
  - ```{autodoc2-docstring} src.policies.other.post_processing.branch_and_price.BranchAndPricePostProcessor
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_time_limit <src.policies.other.post_processing.branch_and_price._time_limit>`
  - ```{autodoc2-docstring} src.policies.other.post_processing.branch_and_price._time_limit
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.other.post_processing.branch_and_price.logger>`
  - ```{autodoc2-docstring} src.policies.other.post_processing.branch_and_price.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.other.post_processing.branch_and_price.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.other.post_processing.branch_and_price.logger
```

````

````{py:function} _time_limit(seconds: float)
:canonical: src.policies.other.post_processing.branch_and_price._time_limit

```{autodoc2-docstring} src.policies.other.post_processing.branch_and_price._time_limit
```
````

`````{py:class} BranchAndPricePostProcessor(**kwargs: typing.Any)
:canonical: src.policies.other.post_processing.branch_and_price.BranchAndPricePostProcessor

Bases: {py:obj}`logic.src.interfaces.post_processing.IPostProcessor`

```{autodoc2-docstring} src.policies.other.post_processing.branch_and_price.BranchAndPricePostProcessor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.post_processing.branch_and_price.BranchAndPricePostProcessor.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.other.post_processing.branch_and_price.BranchAndPricePostProcessor.process

````

````{py:method} _solve_inhouse(input_routes: typing.List[typing.List[int]], dm: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, cost_per_km: float, revenue_kg: float, mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Optional[typing.List[typing.List[int]]]
:canonical: src.policies.other.post_processing.branch_and_price.BranchAndPricePostProcessor._solve_inhouse

```{autodoc2-docstring} src.policies.other.post_processing.branch_and_price.BranchAndPricePostProcessor._solve_inhouse
```

````

````{py:method} _solve_vrpy(input_routes: typing.List[typing.List[int]], dm: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, cost_per_km: float, revenue_kg: float, mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Optional[typing.List[typing.List[int]]]
:canonical: src.policies.other.post_processing.branch_and_price.BranchAndPricePostProcessor._solve_vrpy

```{autodoc2-docstring} src.policies.other.post_processing.branch_and_price.BranchAndPricePostProcessor._solve_vrpy
```

````

````{py:method} _fallback_set_partitioning(tour: typing.List[int], **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.other.post_processing.branch_and_price.BranchAndPricePostProcessor._fallback_set_partitioning

```{autodoc2-docstring} src.policies.other.post_processing.branch_and_price.BranchAndPricePostProcessor._fallback_set_partitioning
```

````

`````
