# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_LinUCBArm <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._LinUCBArm>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._LinUCBArm
    :summary:
    ```
* - {py:obj}`_SlidingWindowStats <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._SlidingWindowStats>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._SlidingWindowStats
    :summary:
    ```
* - {py:obj}`RLController <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_build_context_vector <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._build_context_vector>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._build_context_vector
    :summary:
    ```
* - {py:obj}`_action_to_budget_fracs <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._action_to_budget_fracs>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._action_to_budget_fracs
    :summary:
    ```
* - {py:obj}`_action_to_operator_multipliers <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._action_to_operator_multipliers>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._action_to_operator_multipliers
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.logger
    :summary:
    ```
* - {py:obj}`_STATE_DEFAULTS <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._STATE_DEFAULTS>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._STATE_DEFAULTS
    :summary:
    ```
* - {py:obj}`_BUDGET_GRID <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._BUDGET_GRID>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._BUDGET_GRID
    :summary:
    ```
* - {py:obj}`_OPERATOR_GRID <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._OPERATOR_GRID>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._OPERATOR_GRID
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.logger
```

````

````{py:data} _STATE_DEFAULTS
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._STATE_DEFAULTS
:type: typing.Dict[str, float]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._STATE_DEFAULTS
```

````

````{py:function} _build_context_vector(features: typing.List[str], values: typing.Dict[str, float]) -> numpy.ndarray
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._build_context_vector

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._build_context_vector
```
````

````{py:data} _BUDGET_GRID
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._BUDGET_GRID
:value: >
   [0.05, 0.1, 0.2, 0.35, 0.5]

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._BUDGET_GRID
```

````

````{py:data} _OPERATOR_GRID
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._OPERATOR_GRID
:value: >
   [0.5, 0.8, 1.0, 1.5, 2.0]

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._OPERATOR_GRID
```

````

````{py:function} _action_to_budget_fracs(action: numpy.ndarray) -> typing.Dict[str, float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._action_to_budget_fracs

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._action_to_budget_fracs
```
````

````{py:function} _action_to_operator_multipliers(action: numpy.ndarray) -> numpy.ndarray
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._action_to_operator_multipliers

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._action_to_operator_multipliers
```
````

`````{py:class} _LinUCBArm(d: int, lam: float = 1.0)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._LinUCBArm

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._LinUCBArm
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._LinUCBArm.__init__
```

````{py:method} theta() -> numpy.ndarray
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._LinUCBArm.theta

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._LinUCBArm.theta
```

````

````{py:method} ucb(x: numpy.ndarray, alpha: float) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._LinUCBArm.ucb

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._LinUCBArm.ucb
```

````

````{py:method} update(x: numpy.ndarray, reward: float) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._LinUCBArm.update

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._LinUCBArm.update
```

````

````{py:method} to_dict() -> typing.Dict
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._LinUCBArm.to_dict

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._LinUCBArm.to_dict
```

````

````{py:method} from_dict(d: int, data: typing.Dict) -> src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._LinUCBArm
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._LinUCBArm.from_dict
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._LinUCBArm.from_dict
```

````

`````

`````{py:class} _SlidingWindowStats(n_arms: int, window: int)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._SlidingWindowStats

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._SlidingWindowStats
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._SlidingWindowStats.__init__
```

````{py:method} update(arm: int, reward: float) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._SlidingWindowStats.update

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._SlidingWindowStats.update
```

````

````{py:method} mean(arm: int) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._SlidingWindowStats.mean

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller._SlidingWindowStats.mean
```

````

`````

`````{py:class} RLController(params: typing.Any)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController.__init__
```

````{py:method} make_context(n_nodes: int, fill_levels: numpy.ndarray, mandatory_ratio: float, lp_ub: float, best_profit: float, pool_size: int, time_remaining: float, time_total: float) -> numpy.ndarray
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController.make_context

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController.make_context
```

````

````{py:method} act(context: numpy.ndarray, time_limit: float, budget_override_from_alpha: typing.Optional[typing.Dict[str, float]] = None) -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController.act

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController.act
```

````

````{py:method} update(context: numpy.ndarray, action_levels: typing.List[int], delta_profit: float, delta_time: float, best_profit: float) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController.update

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController.update
```

````

````{py:method} _shape_reward(delta_profit: float, delta_time: float, best_profit: float) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController._shape_reward

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController._shape_reward
```

````

````{py:method} _default_action(alpha_defaults: typing.Optional[typing.Dict[str, float]]) -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController._default_action

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController._default_action
```

````

````{py:method} save(path: str) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController.save

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController.save
```

````

````{py:method} _load_policy(path: str) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController._load_policy

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.rl_controller.RLController._load_policy
```

````

`````
