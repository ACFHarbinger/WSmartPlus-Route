# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`reset_rl_controller <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher.reset_rl_controller>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher.reset_rl_controller
    :summary:
    ```
* - {py:obj}`_get_rl_controller <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._get_rl_controller>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._get_rl_controller
    :summary:
    ```
* - {py:obj}`_import_alns_stage <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._import_alns_stage>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._import_alns_stage
    :summary:
    ```
* - {py:obj}`_import_bpc_stage <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._import_bpc_stage>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._import_bpc_stage
    :summary:
    ```
* - {py:obj}`_import_sp_stage <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._import_sp_stage>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._import_sp_stage
    :summary:
    ```
* - {py:obj}`_import_alns_params <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._import_alns_params>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._import_alns_params
    :summary:
    ```
* - {py:obj}`_mandatory_local_set <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._mandatory_local_set>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._mandatory_local_set
    :summary:
    ```
* - {py:obj}`run_lasm_pipeline <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher.run_lasm_pipeline>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher.run_lasm_pipeline
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher.logger
    :summary:
    ```
* - {py:obj}`_GLOBAL_RL <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._GLOBAL_RL>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._GLOBAL_RL
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher.logger
```

````

````{py:data} _GLOBAL_RL
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._GLOBAL_RL
:type: typing.Optional[src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._GLOBAL_RL
```

````

````{py:function} reset_rl_controller() -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher.reset_rl_controller

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher.reset_rl_controller
```
````

````{py:function} _get_rl_controller(params: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams) -> src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._get_rl_controller

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._get_rl_controller
```
````

````{py:function} _import_alns_stage() -> typing.Optional[typing.Callable[[typing.Any, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any], typing.Optional[typing.Any]]]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._import_alns_stage

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._import_alns_stage
```
````

````{py:function} _import_bpc_stage() -> typing.Optional[typing.Callable[[typing.Any, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any], typing.Optional[typing.Any]]]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._import_bpc_stage

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._import_bpc_stage
```
````

````{py:function} _import_sp_stage() -> typing.Optional[typing.Callable[[typing.Any, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any], typing.Optional[typing.Any]]]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._import_sp_stage

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._import_sp_stage
```
````

````{py:function} _import_alns_params() -> typing.Optional[typing.Type[typing.Any]]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._import_alns_params

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._import_alns_params
```
````

````{py:function} _mandatory_local_set(bins: numpy.typing.NDArray[numpy.float64], binsids: typing.List[int], mandatory: typing.List[int], psi: float) -> typing.Set[int]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._mandatory_local_set

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher._mandatory_local_set
```
````

````{py:function} run_lasm_pipeline(bins: numpy.typing.NDArray[numpy.float64], dist_matrix: typing.List[typing.List[float]], env: typing.Any, values: typing.Dict[str, typing.Any], binsids: typing.List[int], mandatory: typing.List[int], n_vehicles: int = 1, params: typing.Optional[src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams] = None, recorder: typing.Any = None) -> typing.Tuple[typing.List[int], float, float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher.run_lasm_pipeline

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.dispatcher.run_lasm_pipeline
```
````
