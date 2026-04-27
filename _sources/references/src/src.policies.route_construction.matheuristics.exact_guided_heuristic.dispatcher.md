# {py:mod}`src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher`

```{py:module} src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_build_alns_params <src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher._build_alns_params>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher._build_alns_params
    :summary:
    ```
* - {py:obj}`_mandatory_local_set <src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher._mandatory_local_set>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher._mandatory_local_set
    :summary:
    ```
* - {py:obj}`run_pipeline <src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher.run_pipeline>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher.run_pipeline
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher.logger
```

````

````{py:function} _build_alns_params(pipeline_params: src.policies.route_construction.matheuristics.exact_guided_heuristic.params.PipelineParams, time_limit: float)
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher._build_alns_params

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher._build_alns_params
```
````

````{py:function} _mandatory_local_set(bins: numpy.typing.NDArray[numpy.float64], binsids: typing.List[int], mandatory: typing.List[int], psi: float) -> typing.Set[int]
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher._mandatory_local_set

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher._mandatory_local_set
```
````

````{py:function} run_pipeline(bins: numpy.typing.NDArray[numpy.float64], dist_matrix: typing.List[typing.List[float]], env, values: typing.Dict[str, typing.Any], binsids: typing.List[int], mandatory: typing.List[int], n_vehicles: int = 1, params: typing.Optional[src.policies.route_construction.matheuristics.exact_guided_heuristic.params.PipelineParams] = None, recorder=None) -> typing.Tuple[typing.List[int], float, float]
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher.run_pipeline

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.dispatcher.run_pipeline
```
````
