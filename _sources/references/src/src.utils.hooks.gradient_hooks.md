# {py:mod}`src.utils.hooks.gradient_hooks`

```{py:module} src.utils.hooks.gradient_hooks
```

```{autodoc2-docstring} src.utils.hooks.gradient_hooks
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`add_gradient_monitoring_hooks <src.utils.hooks.gradient_hooks.add_gradient_monitoring_hooks>`
  - ```{autodoc2-docstring} src.utils.hooks.gradient_hooks.add_gradient_monitoring_hooks
    :summary:
    ```
* - {py:obj}`add_gradient_clipping_hook <src.utils.hooks.gradient_hooks.add_gradient_clipping_hook>`
  - ```{autodoc2-docstring} src.utils.hooks.gradient_hooks.add_gradient_clipping_hook
    :summary:
    ```
* - {py:obj}`add_gradient_accumulation_hook <src.utils.hooks.gradient_hooks.add_gradient_accumulation_hook>`
  - ```{autodoc2-docstring} src.utils.hooks.gradient_hooks.add_gradient_accumulation_hook
    :summary:
    ```
* - {py:obj}`add_gradient_nan_detector_hook <src.utils.hooks.gradient_hooks.add_gradient_nan_detector_hook>`
  - ```{autodoc2-docstring} src.utils.hooks.gradient_hooks.add_gradient_nan_detector_hook
    :summary:
    ```
* - {py:obj}`remove_all_hooks <src.utils.hooks.gradient_hooks.remove_all_hooks>`
  - ```{autodoc2-docstring} src.utils.hooks.gradient_hooks.remove_all_hooks
    :summary:
    ```
* - {py:obj}`print_gradient_statistics <src.utils.hooks.gradient_hooks.print_gradient_statistics>`
  - ```{autodoc2-docstring} src.utils.hooks.gradient_hooks.print_gradient_statistics
    :summary:
    ```
````

### API

````{py:function} add_gradient_monitoring_hooks(model: torch.nn.Module, layer_names: typing.Optional[typing.List[str]] = None, gradient_threshold: float = 10.0, verbose: bool = True) -> typing.Dict[str, typing.Any]
:canonical: src.utils.hooks.gradient_hooks.add_gradient_monitoring_hooks

```{autodoc2-docstring} src.utils.hooks.gradient_hooks.add_gradient_monitoring_hooks
```
````

````{py:function} add_gradient_clipping_hook(model: torch.nn.Module, max_norm: float = 1.0, layer_names: typing.Optional[typing.List[str]] = None) -> typing.List[typing.Any]
:canonical: src.utils.hooks.gradient_hooks.add_gradient_clipping_hook

```{autodoc2-docstring} src.utils.hooks.gradient_hooks.add_gradient_clipping_hook
```
````

````{py:function} add_gradient_accumulation_hook(model: torch.nn.Module, accumulation_steps: int = 4) -> typing.Dict[str, typing.Any]
:canonical: src.utils.hooks.gradient_hooks.add_gradient_accumulation_hook

```{autodoc2-docstring} src.utils.hooks.gradient_hooks.add_gradient_accumulation_hook
```
````

````{py:function} add_gradient_nan_detector_hook(model: torch.nn.Module, raise_on_nan: bool = True) -> typing.List[typing.Any]
:canonical: src.utils.hooks.gradient_hooks.add_gradient_nan_detector_hook

```{autodoc2-docstring} src.utils.hooks.gradient_hooks.add_gradient_nan_detector_hook
```
````

````{py:function} remove_all_hooks(hook_data: typing.Dict[str, typing.Any]) -> None
:canonical: src.utils.hooks.gradient_hooks.remove_all_hooks

```{autodoc2-docstring} src.utils.hooks.gradient_hooks.remove_all_hooks
```
````

````{py:function} print_gradient_statistics(gradient_stats: typing.List[typing.Dict[str, typing.Any]], top_k: int = 10) -> None
:canonical: src.utils.hooks.gradient_hooks.print_gradient_statistics

```{autodoc2-docstring} src.utils.hooks.gradient_hooks.print_gradient_statistics
```
````
