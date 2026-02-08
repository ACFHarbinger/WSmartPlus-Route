# {py:mod}`src.models.subnets.modules.tsp_adapter`

```{py:module} src.models.subnets.modules.tsp_adapter
```

```{autodoc2-docstring} src.models.subnets.modules.tsp_adapter
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TSPAdapter <src.models.subnets.modules.tsp_adapter.TSPAdapter>`
  - ```{autodoc2-docstring} src.models.subnets.modules.tsp_adapter.TSPAdapter
    :summary:
    ```
````

### API

`````{py:class} TSPAdapter(td: tensordict.TensorDict, partition_actions: torch.Tensor, subprob_batch_size: int = 2000, **kwargs)
:canonical: src.models.subnets.modules.tsp_adapter.TSPAdapter

Bases: {py:obj}`src.models.subnets.modules.adapter_base.SubproblemAdapter`

```{autodoc2-docstring} src.models.subnets.modules.tsp_adapter.TSPAdapter
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.tsp_adapter.TSPAdapter.__init__
```

````{py:attribute} subproblem_env_name
:canonical: src.models.subnets.modules.tsp_adapter.TSPAdapter.subproblem_env_name
:type: str
:value: >
   'tsp'

```{autodoc2-docstring} src.models.subnets.modules.tsp_adapter.TSPAdapter.subproblem_env_name
```

````

````{py:method} _extract_subproblems() -> None
:canonical: src.models.subnets.modules.tsp_adapter.TSPAdapter._extract_subproblems

```{autodoc2-docstring} src.models.subnets.modules.tsp_adapter.TSPAdapter._extract_subproblems
```

````

````{py:method} get_batched_subprobs() -> typing.Iterator[src.models.subnets.modules.subproblem_mapping.SubproblemMapping]
:canonical: src.models.subnets.modules.tsp_adapter.TSPAdapter.get_batched_subprobs

```{autodoc2-docstring} src.models.subnets.modules.tsp_adapter.TSPAdapter.get_batched_subprobs
```

````

````{py:method} update_actions(mapping: src.models.subnets.modules.subproblem_mapping.SubproblemMapping, subprob_actions: torch.Tensor) -> None
:canonical: src.models.subnets.modules.tsp_adapter.TSPAdapter.update_actions

```{autodoc2-docstring} src.models.subnets.modules.tsp_adapter.TSPAdapter.update_actions
```

````

`````
