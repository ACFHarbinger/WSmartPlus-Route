# {py:mod}`src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.policy_kgls`

```{py:module} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.policy_kgls
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.policy_kgls
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KGLSPolicy <src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.policy_kgls.KGLSPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.policy_kgls.KGLSPolicy
    :summary:
    ```
````

### API

`````{py:class} KGLSPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.KGLSConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.policy_kgls.KGLSPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.policy_kgls.KGLSPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.policy_kgls.KGLSPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.policy_kgls.KGLSPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.policy_kgls.KGLSPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.policy_kgls.KGLSPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.policy_kgls.KGLSPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.policy_kgls.KGLSPolicy._run_solver
```

````

`````
