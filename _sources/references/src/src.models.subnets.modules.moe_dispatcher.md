# {py:mod}`src.models.subnets.modules.moe_dispatcher`

```{py:module} src.models.subnets.modules.moe_dispatcher
```

```{autodoc2-docstring} src.models.subnets.modules.moe_dispatcher
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SparseDispatcher <src.models.subnets.modules.moe_dispatcher.SparseDispatcher>`
  - ```{autodoc2-docstring} src.models.subnets.modules.moe_dispatcher.SparseDispatcher
    :summary:
    ```
````

### API

`````{py:class} SparseDispatcher(num_experts: int, gates: torch.Tensor)
:canonical: src.models.subnets.modules.moe_dispatcher.SparseDispatcher

```{autodoc2-docstring} src.models.subnets.modules.moe_dispatcher.SparseDispatcher
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.moe_dispatcher.SparseDispatcher.__init__
```

````{py:method} dispatch(inp: torch.Tensor) -> typing.List[torch.Tensor]
:canonical: src.models.subnets.modules.moe_dispatcher.SparseDispatcher.dispatch

```{autodoc2-docstring} src.models.subnets.modules.moe_dispatcher.SparseDispatcher.dispatch
```

````

````{py:method} combine(expert_out: typing.List[torch.Tensor], multiply_by_gates: bool = True) -> torch.Tensor
:canonical: src.models.subnets.modules.moe_dispatcher.SparseDispatcher.combine

```{autodoc2-docstring} src.models.subnets.modules.moe_dispatcher.SparseDispatcher.combine
```

````

````{py:method} expert_to_gates() -> typing.List[torch.Tensor]
:canonical: src.models.subnets.modules.moe_dispatcher.SparseDispatcher.expert_to_gates

```{autodoc2-docstring} src.models.subnets.modules.moe_dispatcher.SparseDispatcher.expert_to_gates
```

````

`````
