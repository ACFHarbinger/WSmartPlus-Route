# {py:mod}`src.tracking.hooks`

```{py:module} src.tracking.hooks
```

```{autodoc2-docstring} src.tracking.hooks
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.tracking.hooks.weight_hooks
src.tracking.hooks.gradient_hooks
src.tracking.hooks.memory_hooks
src.tracking.hooks.attention_hooks
src.tracking.hooks.activation_hooks
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`register_hooks_with_run <src.tracking.hooks.register_hooks_with_run>`
  - ```{autodoc2-docstring} src.tracking.hooks.register_hooks_with_run
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.tracking.hooks.__all__>`
  - ```{autodoc2-docstring} src.tracking.hooks.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: src.tracking.hooks.__all__
:value: >
   ['add_attention_hooks', 'add_gradient_monitoring_hooks', 'add_gradient_clipping_hook', 'add_gradient...

```{autodoc2-docstring} src.tracking.hooks.__all__
```

````

````{py:function} register_hooks_with_run(hook_data: typing.Dict[str, typing.Any], run: typing.Any, prefix: str = 'hooks') -> None
:canonical: src.tracking.hooks.register_hooks_with_run

```{autodoc2-docstring} src.tracking.hooks.register_hooks_with_run
```
````
