# {py:mod}`src.policies.selection_and_construction.base.base_joint_policy`

```{py:module} src.policies.selection_and_construction.base.base_joint_policy
```

```{autodoc2-docstring} src.policies.selection_and_construction.base.base_joint_policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BaseJointPolicy <src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy
    :summary:
    ```
````

### API

`````{py:class} BaseJointPolicy(config: typing.Any = None)
:canonical: src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`, {py:obj}`logic.src.interfaces.route_constructor.IRouteConstructor`

```{autodoc2-docstring} src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy.__init__
```

````{py:property} config
:canonical: src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy.config
:type: typing.Any

```{autodoc2-docstring} src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy.config
```

````

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy._get_config_key

```{autodoc2-docstring} src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy._get_config_key
```

````

````{py:method} solve_joint(context: logic.src.interfaces.context.joint_context.JointSelectionConstructionContext) -> typing.Tuple[typing.List[int], typing.List[typing.List[int]], float, float]
:canonical: src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy.solve_joint
:abstractmethod:

```{autodoc2-docstring} src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy.solve_joint
```

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, float, typing.Optional[logic.src.interfaces.context.search_context.SearchContext], typing.Optional[typing.Any]]
:canonical: src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy.execute

```{autodoc2-docstring} src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy.execute
```

````

````{py:method} _build_subset_problem(selected_bins: typing.List[int], context: logic.src.interfaces.context.joint_context.JointSelectionConstructionContext) -> typing.Tuple[numpy.ndarray, typing.Dict[int, float], typing.List[int]]
:canonical: src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy._build_subset_problem

```{autodoc2-docstring} src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy._build_subset_problem
```

````

````{py:method} _routes_to_flat_tour(routes: typing.List[typing.List[int]], subset_indices: typing.List[int]) -> typing.List[int]
:canonical: src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy._routes_to_flat_tour
:staticmethod:

```{autodoc2-docstring} src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy._routes_to_flat_tour
```

````

`````
