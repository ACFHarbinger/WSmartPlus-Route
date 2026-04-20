# {py:mod}`src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa`

```{py:module} src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa
```

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointSAPolicy <src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy
    :summary:
    ```
````

### API

`````{py:class} JointSAPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.jsa.JointSAConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy

Bases: {py:obj}`logic.src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy`

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy._get_config_key

````

````{py:method} solve_joint(context: logic.src.interfaces.context.joint_context.JointSelectionConstructionContext) -> typing.Tuple[typing.List[int], typing.List[typing.List[int]], float, float]
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy.solve_joint

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy.solve_joint
```

````

````{py:method} _initial_selection(context: logic.src.interfaces.context.joint_context.JointSelectionConstructionContext) -> typing.List[int]
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy._initial_selection

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy._initial_selection
```

````

````{py:method} _construct_greedy_routes(selected_bins: typing.List[int], context: logic.src.interfaces.context.joint_context.JointSelectionConstructionContext) -> typing.List[typing.List[int]]
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy._construct_greedy_routes

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy._construct_greedy_routes
```

````

````{py:method} _evaluate(selected_bins: typing.List[int], routes: typing.List[typing.List[int]], context: logic.src.interfaces.context.joint_context.JointSelectionConstructionContext, params: src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams) -> typing.Tuple[float, float, float]
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy._evaluate

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy._evaluate
```

````

````{py:method} _total_overflow_penalty(selected_bins: typing.List[int], context: logic.src.interfaces.context.joint_context.JointSelectionConstructionContext, params: src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams) -> float
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy._total_overflow_penalty

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy._total_overflow_penalty
```

````

````{py:method} _generate_neighbor(current_selection: typing.List[int], current_routes: typing.List[typing.List[int]], context: logic.src.interfaces.context.joint_context.JointSelectionConstructionContext, params: src.policies.selection_and_construction.joint_simulated_annealing.params.JointSAParams) -> typing.Tuple[typing.List[int], typing.List[typing.List[int]]]
:canonical: src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy._generate_neighbor

```{autodoc2-docstring} src.policies.selection_and_construction.joint_simulated_annealing.policy_jsa.JointSAPolicy._generate_neighbor
```

````

`````
