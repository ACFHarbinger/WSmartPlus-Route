# {py:mod}`src.policies.exact_stochastic_dynamic_programming.policy_esdp`

```{py:module} src.policies.exact_stochastic_dynamic_programming.policy_esdp
```

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.policy_esdp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ExactSDPPolicy <src.policies.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy>`
  - ```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_SDP_CACHE <src.policies.exact_stochastic_dynamic_programming.policy_esdp._SDP_CACHE>`
  - ```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.policy_esdp._SDP_CACHE
    :summary:
    ```
````

### API

````{py:data} _SDP_CACHE
:canonical: src.policies.exact_stochastic_dynamic_programming.policy_esdp._SDP_CACHE
:value: >
   None

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.policy_esdp._SDP_CACHE
```

````

`````{py:class} ExactSDPPolicy(config: typing.Any = None)
:canonical: src.policies.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy.__init__
```

````{py:method} _get_config_class()
:canonical: src.policies.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy._get_config_class

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy._get_config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy._get_config_key

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy.execute

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.policy_esdp.ExactSDPPolicy.execute
```

````

`````
