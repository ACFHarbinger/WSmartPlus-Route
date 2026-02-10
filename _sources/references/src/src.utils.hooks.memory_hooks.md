# {py:mod}`src.utils.hooks.memory_hooks`

```{py:module} src.utils.hooks.memory_hooks
```

```{autodoc2-docstring} src.utils.hooks.memory_hooks
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`add_memory_profiling_hooks <src.utils.hooks.memory_hooks.add_memory_profiling_hooks>`
  - ```{autodoc2-docstring} src.utils.hooks.memory_hooks.add_memory_profiling_hooks
    :summary:
    ```
* - {py:obj}`add_memory_leak_detector_hook <src.utils.hooks.memory_hooks.add_memory_leak_detector_hook>`
  - ```{autodoc2-docstring} src.utils.hooks.memory_hooks.add_memory_leak_detector_hook
    :summary:
    ```
* - {py:obj}`estimate_model_memory <src.utils.hooks.memory_hooks.estimate_model_memory>`
  - ```{autodoc2-docstring} src.utils.hooks.memory_hooks.estimate_model_memory
    :summary:
    ```
* - {py:obj}`print_memory_summary <src.utils.hooks.memory_hooks.print_memory_summary>`
  - ```{autodoc2-docstring} src.utils.hooks.memory_hooks.print_memory_summary
    :summary:
    ```
* - {py:obj}`optimize_batch_size <src.utils.hooks.memory_hooks.optimize_batch_size>`
  - ```{autodoc2-docstring} src.utils.hooks.memory_hooks.optimize_batch_size
    :summary:
    ```
* - {py:obj}`remove_all_hooks <src.utils.hooks.memory_hooks.remove_all_hooks>`
  - ```{autodoc2-docstring} src.utils.hooks.memory_hooks.remove_all_hooks
    :summary:
    ```
````

### API

````{py:function} add_memory_profiling_hooks(model: torch.nn.Module, device: typing.Optional[torch.device] = None, track_allocations: bool = True) -> typing.Dict[str, typing.Any]
:canonical: src.utils.hooks.memory_hooks.add_memory_profiling_hooks

```{autodoc2-docstring} src.utils.hooks.memory_hooks.add_memory_profiling_hooks
```
````

````{py:function} add_memory_leak_detector_hook(model: torch.nn.Module, threshold_mb: float = 100.0, device: typing.Optional[torch.device] = None) -> typing.Dict[str, typing.Any]
:canonical: src.utils.hooks.memory_hooks.add_memory_leak_detector_hook

```{autodoc2-docstring} src.utils.hooks.memory_hooks.add_memory_leak_detector_hook
```
````

````{py:function} estimate_model_memory(model: torch.nn.Module, input_shape: tuple, batch_size: int = 1, dtype: torch.dtype = torch.float32) -> typing.Dict[str, float]
:canonical: src.utils.hooks.memory_hooks.estimate_model_memory

```{autodoc2-docstring} src.utils.hooks.memory_hooks.estimate_model_memory
```
````

````{py:function} print_memory_summary(memory_stats: typing.List[typing.Dict[str, typing.Any]], top_k: int = 10) -> None
:canonical: src.utils.hooks.memory_hooks.print_memory_summary

```{autodoc2-docstring} src.utils.hooks.memory_hooks.print_memory_summary
```
````

````{py:function} optimize_batch_size(model: torch.nn.Module, input_generator: typing.Callable, initial_batch_size: int = 32, device: typing.Optional[torch.device] = None, safety_margin: float = 0.9) -> int
:canonical: src.utils.hooks.memory_hooks.optimize_batch_size

```{autodoc2-docstring} src.utils.hooks.memory_hooks.optimize_batch_size
```
````

````{py:function} remove_all_hooks(hook_data: typing.Dict[str, typing.Any]) -> None
:canonical: src.utils.hooks.memory_hooks.remove_all_hooks

```{autodoc2-docstring} src.utils.hooks.memory_hooks.remove_all_hooks
```
````
