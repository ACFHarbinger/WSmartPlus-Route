# {py:mod}`src.pipeline.rl.meta.registry`

```{py:module} src.pipeline.rl.meta.registry
```

```{autodoc2-docstring} src.pipeline.rl.meta.registry
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_meta_strategy <src.pipeline.rl.meta.registry.get_meta_strategy>`
  - ```{autodoc2-docstring} src.pipeline.rl.meta.registry.get_meta_strategy
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`META_STRATEGY_REGISTRY <src.pipeline.rl.meta.registry.META_STRATEGY_REGISTRY>`
  - ```{autodoc2-docstring} src.pipeline.rl.meta.registry.META_STRATEGY_REGISTRY
    :summary:
    ```
````

### API

````{py:data} META_STRATEGY_REGISTRY
:canonical: src.pipeline.rl.meta.registry.META_STRATEGY_REGISTRY
:type: typing.Dict[str, typing.Type[logic.src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy]]
:value: >
   None

```{autodoc2-docstring} src.pipeline.rl.meta.registry.META_STRATEGY_REGISTRY
```

````

````{py:function} get_meta_strategy(name: str, **kwargs) -> logic.src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy
:canonical: src.pipeline.rl.meta.registry.get_meta_strategy

```{autodoc2-docstring} src.pipeline.rl.meta.registry.get_meta_strategy
```
````
