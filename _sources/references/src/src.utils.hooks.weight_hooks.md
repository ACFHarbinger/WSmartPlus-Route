# {py:mod}`src.utils.hooks.weight_hooks`

```{py:module} src.utils.hooks.weight_hooks
```

```{autodoc2-docstring} src.utils.hooks.weight_hooks
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`add_weight_change_monitor_hook <src.utils.hooks.weight_hooks.add_weight_change_monitor_hook>`
  - ```{autodoc2-docstring} src.utils.hooks.weight_hooks.add_weight_change_monitor_hook
    :summary:
    ```
* - {py:obj}`compute_weight_changes <src.utils.hooks.weight_hooks.compute_weight_changes>`
  - ```{autodoc2-docstring} src.utils.hooks.weight_hooks.compute_weight_changes
    :summary:
    ```
* - {py:obj}`add_weight_distribution_monitor <src.utils.hooks.weight_hooks.add_weight_distribution_monitor>`
  - ```{autodoc2-docstring} src.utils.hooks.weight_hooks.add_weight_distribution_monitor
    :summary:
    ```
* - {py:obj}`add_weight_update_monitor_hook <src.utils.hooks.weight_hooks.add_weight_update_monitor_hook>`
  - ```{autodoc2-docstring} src.utils.hooks.weight_hooks.add_weight_update_monitor_hook
    :summary:
    ```
* - {py:obj}`restore_optimizer_step <src.utils.hooks.weight_hooks.restore_optimizer_step>`
  - ```{autodoc2-docstring} src.utils.hooks.weight_hooks.restore_optimizer_step
    :summary:
    ```
* - {py:obj}`add_weight_norm_constraint_hook <src.utils.hooks.weight_hooks.add_weight_norm_constraint_hook>`
  - ```{autodoc2-docstring} src.utils.hooks.weight_hooks.add_weight_norm_constraint_hook
    :summary:
    ```
* - {py:obj}`detect_weight_symmetry_breaking <src.utils.hooks.weight_hooks.detect_weight_symmetry_breaking>`
  - ```{autodoc2-docstring} src.utils.hooks.weight_hooks.detect_weight_symmetry_breaking
    :summary:
    ```
* - {py:obj}`print_weight_summary <src.utils.hooks.weight_hooks.print_weight_summary>`
  - ```{autodoc2-docstring} src.utils.hooks.weight_hooks.print_weight_summary
    :summary:
    ```
* - {py:obj}`analyze_weight_updates <src.utils.hooks.weight_hooks.analyze_weight_updates>`
  - ```{autodoc2-docstring} src.utils.hooks.weight_hooks.analyze_weight_updates
    :summary:
    ```
````

### API

````{py:function} add_weight_change_monitor_hook(model: torch.nn.Module, layer_names: typing.Optional[typing.List[str]] = None) -> typing.Dict[str, typing.Any]
:canonical: src.utils.hooks.weight_hooks.add_weight_change_monitor_hook

```{autodoc2-docstring} src.utils.hooks.weight_hooks.add_weight_change_monitor_hook
```
````

````{py:function} compute_weight_changes(model: torch.nn.Module, hook_data: typing.Dict[str, typing.Any], metric: str = 'norm') -> typing.Dict[str, float]
:canonical: src.utils.hooks.weight_hooks.compute_weight_changes

```{autodoc2-docstring} src.utils.hooks.weight_hooks.compute_weight_changes
```
````

````{py:function} add_weight_distribution_monitor(model: torch.nn.Module, layer_types: typing.Tuple[type, ...] = (nn.Linear, nn.Conv2d)) -> typing.Dict[str, typing.Any]
:canonical: src.utils.hooks.weight_hooks.add_weight_distribution_monitor

```{autodoc2-docstring} src.utils.hooks.weight_hooks.add_weight_distribution_monitor
```
````

````{py:function} add_weight_update_monitor_hook(optimizer: torch.optim.Optimizer, log_interval: int = 10) -> typing.Dict[str, typing.Any]
:canonical: src.utils.hooks.weight_hooks.add_weight_update_monitor_hook

```{autodoc2-docstring} src.utils.hooks.weight_hooks.add_weight_update_monitor_hook
```
````

````{py:function} restore_optimizer_step(optimizer: torch.optim.Optimizer, hook_data: typing.Dict[str, typing.Any]) -> None
:canonical: src.utils.hooks.weight_hooks.restore_optimizer_step

```{autodoc2-docstring} src.utils.hooks.weight_hooks.restore_optimizer_step
```
````

````{py:function} add_weight_norm_constraint_hook(model: torch.nn.Module, max_norm: float = 10.0, layer_names: typing.Optional[typing.List[str]] = None) -> typing.List[typing.Any]
:canonical: src.utils.hooks.weight_hooks.add_weight_norm_constraint_hook

```{autodoc2-docstring} src.utils.hooks.weight_hooks.add_weight_norm_constraint_hook
```
````

````{py:function} detect_weight_symmetry_breaking(model: torch.nn.Module, threshold: float = 0.0001) -> typing.Dict[str, bool]
:canonical: src.utils.hooks.weight_hooks.detect_weight_symmetry_breaking

```{autodoc2-docstring} src.utils.hooks.weight_hooks.detect_weight_symmetry_breaking
```
````

````{py:function} print_weight_summary(weight_changes: typing.Dict[str, float], weight_stats: typing.Dict[str, typing.Dict[str, float]], top_k: int = 10) -> None
:canonical: src.utils.hooks.weight_hooks.print_weight_summary

```{autodoc2-docstring} src.utils.hooks.weight_hooks.print_weight_summary
```
````

````{py:function} analyze_weight_updates(update_history: typing.Dict[str, typing.List[float]], window_size: int = 10) -> typing.Dict[str, typing.Dict[str, float]]
:canonical: src.utils.hooks.weight_hooks.analyze_weight_updates

```{autodoc2-docstring} src.utils.hooks.weight_hooks.analyze_weight_updates
```
````
