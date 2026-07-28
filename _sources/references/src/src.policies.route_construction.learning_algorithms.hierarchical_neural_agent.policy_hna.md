# {py:mod}`src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna`

```{py:module} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna
```

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HierarchicalNeuralAgentPolicy <src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna.HierarchicalNeuralAgentPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna.HierarchicalNeuralAgentPolicy
    :summary:
    ```
````

### API

`````{py:class} HierarchicalNeuralAgentPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.HNAPolicyConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna.HierarchicalNeuralAgentPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna.HierarchicalNeuralAgentPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna.HierarchicalNeuralAgentPolicy.__init__
```

````{py:method} _config_class() -> typing.Type[logic.src.configs.policies.HNAPolicyConfig]
:canonical: src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna.HierarchicalNeuralAgentPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna.HierarchicalNeuralAgentPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna.HierarchicalNeuralAgentPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna.HierarchicalNeuralAgentPolicy._get_config_key
```

````

````{py:method} _load_module() -> None
:canonical: src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna.HierarchicalNeuralAgentPolicy._load_module

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna.HierarchicalNeuralAgentPolicy._load_module
```

````

````{py:method} _select_mandatory_nodes(wastes: typing.Dict[int, float], locs: typing.Optional[numpy.ndarray] = None) -> typing.List[int]
:canonical: src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna.HierarchicalNeuralAgentPolicy._select_mandatory_nodes

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna.HierarchicalNeuralAgentPolicy._select_mandatory_nodes
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna.HierarchicalNeuralAgentPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.policy_hna.HierarchicalNeuralAgentPolicy._run_multi_period_solver
```

````

`````
