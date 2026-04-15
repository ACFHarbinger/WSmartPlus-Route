# {py:mod}`src.policies.scenario_tree_extensive_form.policy_st_ef`

```{py:module} src.policies.scenario_tree_extensive_form.policy_st_ef
```

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.policy_st_ef
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ScenarioTreeExtensiveFormPolicy <src.policies.scenario_tree_extensive_form.policy_st_ef.ScenarioTreeExtensiveFormPolicy>`
  - ```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.policy_st_ef.ScenarioTreeExtensiveFormPolicy
    :summary:
    ```
````

### API

`````{py:class} ScenarioTreeExtensiveFormPolicy(config: typing.Any = None)
:canonical: src.policies.scenario_tree_extensive_form.policy_st_ef.ScenarioTreeExtensiveFormPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.policy_st_ef.ScenarioTreeExtensiveFormPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.policy_st_ef.ScenarioTreeExtensiveFormPolicy.__init__
```

````{py:method} _config_class()
:canonical: src.policies.scenario_tree_extensive_form.policy_st_ef.ScenarioTreeExtensiveFormPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.scenario_tree_extensive_form.policy_st_ef.ScenarioTreeExtensiveFormPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.scenario_tree_extensive_form.policy_st_ef.ScenarioTreeExtensiveFormPolicy._run_solver

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.policy_st_ef.ScenarioTreeExtensiveFormPolicy._run_solver
```

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.scenario_tree_extensive_form.policy_st_ef.ScenarioTreeExtensiveFormPolicy.execute

```{autodoc2-docstring} src.policies.scenario_tree_extensive_form.policy_st_ef.ScenarioTreeExtensiveFormPolicy.execute
```

````

`````
