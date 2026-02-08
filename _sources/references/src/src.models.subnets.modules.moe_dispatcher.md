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

`````{py:class} SparseDispatcher(num_experts, gates)
:canonical: src.models.subnets.modules.moe_dispatcher.SparseDispatcher

Bases: {py:obj}`object`

```{autodoc2-docstring} src.models.subnets.modules.moe_dispatcher.SparseDispatcher
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.moe_dispatcher.SparseDispatcher.__init__
```

````{py:method} dispatch(inp)
:canonical: src.models.subnets.modules.moe_dispatcher.SparseDispatcher.dispatch

```{autodoc2-docstring} src.models.subnets.modules.moe_dispatcher.SparseDispatcher.dispatch
```

````

````{py:method} combine(expert_out, multiply_by_gates=True)
:canonical: src.models.subnets.modules.moe_dispatcher.SparseDispatcher.combine

```{autodoc2-docstring} src.models.subnets.modules.moe_dispatcher.SparseDispatcher.combine
```

````

````{py:method} expert_to_gates()
:canonical: src.models.subnets.modules.moe_dispatcher.SparseDispatcher.expert_to_gates

```{autodoc2-docstring} src.models.subnets.modules.moe_dispatcher.SparseDispatcher.expert_to_gates
```

````

`````
