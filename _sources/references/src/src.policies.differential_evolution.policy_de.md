# {py:mod}`src.policies.differential_evolution.policy_de`

```{py:module} src.policies.differential_evolution.policy_de
```

```{autodoc2-docstring} src.policies.differential_evolution.policy_de
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DEPolicyAdapter <src.policies.differential_evolution.policy_de.DEPolicyAdapter>`
  - ```{autodoc2-docstring} src.policies.differential_evolution.policy_de.DEPolicyAdapter
    :summary:
    ```
````

### API

`````{py:class} DEPolicyAdapter(params: typing.Optional[src.policies.differential_evolution.params.DEParams] = None)
:canonical: src.policies.differential_evolution.policy_de.DEPolicyAdapter

Bases: {py:obj}`logic.src.interfaces.IPolicyAdapter`

```{autodoc2-docstring} src.policies.differential_evolution.policy_de.DEPolicyAdapter
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.differential_evolution.policy_de.DEPolicyAdapter.__init__
```

````{py:method} solve(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.differential_evolution.policy_de.DEPolicyAdapter.solve

```{autodoc2-docstring} src.policies.differential_evolution.policy_de.DEPolicyAdapter.solve
```

````

`````
