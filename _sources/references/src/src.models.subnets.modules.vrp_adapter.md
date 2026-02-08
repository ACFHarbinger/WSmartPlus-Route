# {py:mod}`src.models.subnets.modules.vrp_adapter`

```{py:module} src.models.subnets.modules.vrp_adapter
```

```{autodoc2-docstring} src.models.subnets.modules.vrp_adapter
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VRPAdapter <src.models.subnets.modules.vrp_adapter.VRPAdapter>`
  - ```{autodoc2-docstring} src.models.subnets.modules.vrp_adapter.VRPAdapter
    :summary:
    ```
````

### API

`````{py:class} VRPAdapter(td: tensordict.TensorDict, partition_actions: torch.Tensor, subprob_batch_size: int = 2000, capacity: typing.Optional[float] = None, **kwargs)
:canonical: src.models.subnets.modules.vrp_adapter.VRPAdapter

Bases: {py:obj}`src.models.subnets.modules.adapter_base.SubproblemAdapter`

```{autodoc2-docstring} src.models.subnets.modules.vrp_adapter.VRPAdapter
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.vrp_adapter.VRPAdapter.__init__
```

````{py:attribute} subproblem_env_name
:canonical: src.models.subnets.modules.vrp_adapter.VRPAdapter.subproblem_env_name
:type: str
:value: >
   'cvrp'

```{autodoc2-docstring} src.models.subnets.modules.vrp_adapter.VRPAdapter.subproblem_env_name
```

````

````{py:method} _extract_subproblems() -> None
:canonical: src.models.subnets.modules.vrp_adapter.VRPAdapter._extract_subproblems

```{autodoc2-docstring} src.models.subnets.modules.vrp_adapter.VRPAdapter._extract_subproblems
```

````

````{py:method} get_batched_subprobs() -> typing.Iterator[src.models.subnets.modules.subproblem_mapping.SubproblemMapping]
:canonical: src.models.subnets.modules.vrp_adapter.VRPAdapter.get_batched_subprobs

```{autodoc2-docstring} src.models.subnets.modules.vrp_adapter.VRPAdapter.get_batched_subprobs
```

````

````{py:method} update_actions(mapping: src.models.subnets.modules.subproblem_mapping.SubproblemMapping, subprob_actions: torch.Tensor) -> None
:canonical: src.models.subnets.modules.vrp_adapter.VRPAdapter.update_actions

```{autodoc2-docstring} src.models.subnets.modules.vrp_adapter.VRPAdapter.update_actions
```

````

`````
