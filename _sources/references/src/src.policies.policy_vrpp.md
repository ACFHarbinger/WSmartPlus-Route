# {py:mod}`src.policies.policy_vrpp`

```{py:module} src.policies.policy_vrpp
```

```{autodoc2-docstring} src.policies.policy_vrpp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VRPPPolicy <src.policies.policy_vrpp.VRPPPolicy>`
  - ```{autodoc2-docstring} src.policies.policy_vrpp.VRPPPolicy
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`policy_vrpp <src.policies.policy_vrpp.policy_vrpp>`
  - ```{autodoc2-docstring} src.policies.policy_vrpp.policy_vrpp
    :summary:
    ```
````

### API

````{py:function} policy_vrpp(policy: str, bins_c: numpy.typing.NDArray[numpy.float64], bins_means: numpy.typing.NDArray[numpy.float64], bins_std: numpy.typing.NDArray[numpy.float64], distance_matrix: typing.List[typing.List[float]], model_env: typing.Optional[gurobipy.Env], waste_type: str, area: str, n_vehicles: int, config: typing.Optional[dict] = None) -> typing.Tuple[typing.Optional[typing.List[int]], float, float]
:canonical: src.policies.policy_vrpp.policy_vrpp

```{autodoc2-docstring} src.policies.policy_vrpp.policy_vrpp
```
````

`````{py:class} VRPPPolicy
:canonical: src.policies.policy_vrpp.VRPPPolicy

Bases: {py:obj}`logic.src.policies.adapters.IPolicy`

```{autodoc2-docstring} src.policies.policy_vrpp.VRPPPolicy
```

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Any]
:canonical: src.policies.policy_vrpp.VRPPPolicy.execute

```{autodoc2-docstring} src.policies.policy_vrpp.VRPPPolicy.execute
```

````

`````
