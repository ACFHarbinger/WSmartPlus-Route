# {py:mod}`src.policies.local_branching.policy_lb`

```{py:module} src.policies.local_branching.policy_lb
```

```{autodoc2-docstring} src.policies.local_branching.policy_lb
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LocalBranchingPolicy <src.policies.local_branching.policy_lb.LocalBranchingPolicy>`
  - ```{autodoc2-docstring} src.policies.local_branching.policy_lb.LocalBranchingPolicy
    :summary:
    ```
````

### API

`````{py:class} LocalBranchingPolicy(config: typing.Optional[typing.Dict[str, typing.Any]] = None)
:canonical: src.policies.local_branching.policy_lb.LocalBranchingPolicy

Bases: {py:obj}`src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.local_branching.policy_lb.LocalBranchingPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.local_branching.policy_lb.LocalBranchingPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.local_branching.policy_lb.LocalBranchingPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.local_branching.policy_lb.LocalBranchingPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.local_branching.policy_lb.LocalBranchingPolicy._get_config_key

```{autodoc2-docstring} src.policies.local_branching.policy_lb.LocalBranchingPolicy._get_config_key
```

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Any]
:canonical: src.policies.local_branching.policy_lb.LocalBranchingPolicy.execute

```{autodoc2-docstring} src.policies.local_branching.policy_lb.LocalBranchingPolicy.execute
```

````

`````
