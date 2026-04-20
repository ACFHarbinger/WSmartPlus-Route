# {py:mod}`src.policies.selection_and_construction.joint_greedy_orienteering.policy_jgo`

```{py:module} src.policies.selection_and_construction.joint_greedy_orienteering.policy_jgo
```

```{autodoc2-docstring} src.policies.selection_and_construction.joint_greedy_orienteering.policy_jgo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointGreedyPolicy <src.policies.selection_and_construction.joint_greedy_orienteering.policy_jgo.JointGreedyPolicy>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.joint_greedy_orienteering.policy_jgo.JointGreedyPolicy
    :summary:
    ```
````

### API

`````{py:class} JointGreedyPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.jgo.JointGreedyConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.selection_and_construction.joint_greedy_orienteering.policy_jgo.JointGreedyPolicy

Bases: {py:obj}`logic.src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy`

```{autodoc2-docstring} src.policies.selection_and_construction.joint_greedy_orienteering.policy_jgo.JointGreedyPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.selection_and_construction.joint_greedy_orienteering.policy_jgo.JointGreedyPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.selection_and_construction.joint_greedy_orienteering.policy_jgo.JointGreedyPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.selection_and_construction.joint_greedy_orienteering.policy_jgo.JointGreedyPolicy._get_config_key

````

````{py:method} solve_joint(context: logic.src.interfaces.context.joint_context.JointSelectionConstructionContext) -> typing.Tuple[typing.List[int], typing.List[typing.List[int]], float, float]
:canonical: src.policies.selection_and_construction.joint_greedy_orienteering.policy_jgo.JointGreedyPolicy.solve_joint

```{autodoc2-docstring} src.policies.selection_and_construction.joint_greedy_orienteering.policy_jgo.JointGreedyPolicy.solve_joint
```

````

````{py:method} _single_greedy_run(context: logic.src.interfaces.context.joint_context.JointSelectionConstructionContext, params: src.policies.selection_and_construction.joint_greedy_orienteering.params.JointGreedyParams) -> typing.Tuple[typing.List[int], typing.List[typing.List[int]], float, float]
:canonical: src.policies.selection_and_construction.joint_greedy_orienteering.policy_jgo.JointGreedyPolicy._single_greedy_run

```{autodoc2-docstring} src.policies.selection_and_construction.joint_greedy_orienteering.policy_jgo.JointGreedyPolicy._single_greedy_run
```

````

`````
