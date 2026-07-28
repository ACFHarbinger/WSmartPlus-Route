# {py:mod}`src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga`

```{py:module} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga
```

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NDSBRKGAPolicy <src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga.NDSBRKGAPolicy>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga.NDSBRKGAPolicy
    :summary:
    ```
````

### API

`````{py:class} NDSBRKGAPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.nds_brkga.NDSBRKGAConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga.NDSBRKGAPolicy

Bases: {py:obj}`logic.src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy`

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga.NDSBRKGAPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga.NDSBRKGAPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga.NDSBRKGAPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga.NDSBRKGAPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga.NDSBRKGAPolicy._get_config_key

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga.NDSBRKGAPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga.NDSBRKGAPolicy._run_solver

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga.NDSBRKGAPolicy._run_solver
```

````

````{py:method} solve_joint(context: logic.src.interfaces.context.joint_context.JointSelectionConstructionContext) -> typing.Tuple[typing.List[int], typing.List[typing.List[int]], float, float]
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga.NDSBRKGAPolicy.solve_joint

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga.NDSBRKGAPolicy.solve_joint
```

````

`````
