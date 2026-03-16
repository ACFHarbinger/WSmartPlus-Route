# {py:mod}`src.policies.branch_and_bound.policy_bb`

```{py:module} src.policies.branch_and_bound.policy_bb
```

```{autodoc2-docstring} src.policies.branch_and_bound.policy_bb
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BranchAndBoundPolicy <src.policies.branch_and_bound.policy_bb.BranchAndBoundPolicy>`
  - ```{autodoc2-docstring} src.policies.branch_and_bound.policy_bb.BranchAndBoundPolicy
    :summary:
    ```
````

### API

`````{py:class} BranchAndBoundPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.BBConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.branch_and_bound.policy_bb.BranchAndBoundPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.branch_and_bound.policy_bb.BranchAndBoundPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_bound.policy_bb.BranchAndBoundPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.branch_and_bound.policy_bb.BranchAndBoundPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.branch_and_bound.policy_bb.BranchAndBoundPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.branch_and_bound.policy_bb.BranchAndBoundPolicy._get_config_key

```{autodoc2-docstring} src.policies.branch_and_bound.policy_bb.BranchAndBoundPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.branch_and_bound.policy_bb.BranchAndBoundPolicy._run_solver

```{autodoc2-docstring} src.policies.branch_and_bound.policy_bb.BranchAndBoundPolicy._run_solver
```

````

`````
