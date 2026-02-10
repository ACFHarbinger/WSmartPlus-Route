# {py:mod}`src.utils.hooks.activation_hooks`

```{py:module} src.utils.hooks.activation_hooks
```

```{autodoc2-docstring} src.utils.hooks.activation_hooks
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`add_activation_capture_hooks <src.utils.hooks.activation_hooks.add_activation_capture_hooks>`
  - ```{autodoc2-docstring} src.utils.hooks.activation_hooks.add_activation_capture_hooks
    :summary:
    ```
* - {py:obj}`add_dead_neuron_detector_hook <src.utils.hooks.activation_hooks.add_dead_neuron_detector_hook>`
  - ```{autodoc2-docstring} src.utils.hooks.activation_hooks.add_dead_neuron_detector_hook
    :summary:
    ```
* - {py:obj}`add_activation_statistics_hook <src.utils.hooks.activation_hooks.add_activation_statistics_hook>`
  - ```{autodoc2-docstring} src.utils.hooks.activation_hooks.add_activation_statistics_hook
    :summary:
    ```
* - {py:obj}`compute_activation_statistics <src.utils.hooks.activation_hooks.compute_activation_statistics>`
  - ```{autodoc2-docstring} src.utils.hooks.activation_hooks.compute_activation_statistics
    :summary:
    ```
* - {py:obj}`add_activation_sparsity_hook <src.utils.hooks.activation_hooks.add_activation_sparsity_hook>`
  - ```{autodoc2-docstring} src.utils.hooks.activation_hooks.add_activation_sparsity_hook
    :summary:
    ```
* - {py:obj}`compute_sparsity_percentages <src.utils.hooks.activation_hooks.compute_sparsity_percentages>`
  - ```{autodoc2-docstring} src.utils.hooks.activation_hooks.compute_sparsity_percentages
    :summary:
    ```
* - {py:obj}`print_activation_summary <src.utils.hooks.activation_hooks.print_activation_summary>`
  - ```{autodoc2-docstring} src.utils.hooks.activation_hooks.print_activation_summary
    :summary:
    ```
* - {py:obj}`remove_all_hooks <src.utils.hooks.activation_hooks.remove_all_hooks>`
  - ```{autodoc2-docstring} src.utils.hooks.activation_hooks.remove_all_hooks
    :summary:
    ```
````

### API

````{py:function} add_activation_capture_hooks(model: torch.nn.Module, layer_types: typing.Optional[typing.Tuple[type, ...]] = None, layer_names: typing.Optional[typing.List[str]] = None) -> typing.Dict[str, typing.Any]
:canonical: src.utils.hooks.activation_hooks.add_activation_capture_hooks

```{autodoc2-docstring} src.utils.hooks.activation_hooks.add_activation_capture_hooks
```
````

````{py:function} add_dead_neuron_detector_hook(model: torch.nn.Module, threshold: float = 1e-06, layer_types: typing.Tuple[type, ...] = (nn.Linear, nn.Conv2d)) -> typing.Dict[str, typing.Any]
:canonical: src.utils.hooks.activation_hooks.add_dead_neuron_detector_hook

```{autodoc2-docstring} src.utils.hooks.activation_hooks.add_dead_neuron_detector_hook
```
````

````{py:function} add_activation_statistics_hook(model: torch.nn.Module, layer_types: typing.Tuple[type, ...] = (nn.Linear, nn.Conv2d, nn.MultiheadAttention)) -> typing.Dict[str, typing.Any]
:canonical: src.utils.hooks.activation_hooks.add_activation_statistics_hook

```{autodoc2-docstring} src.utils.hooks.activation_hooks.add_activation_statistics_hook
```
````

````{py:function} compute_activation_statistics(statistics: typing.Dict[str, typing.Dict[str, float]]) -> typing.Dict[str, typing.Dict[str, float]]
:canonical: src.utils.hooks.activation_hooks.compute_activation_statistics

```{autodoc2-docstring} src.utils.hooks.activation_hooks.compute_activation_statistics
```
````

````{py:function} add_activation_sparsity_hook(model: torch.nn.Module, threshold: float = 0.01, layer_types: typing.Tuple[type, ...] = (nn.Linear, nn.Conv2d)) -> typing.Dict[str, typing.Any]
:canonical: src.utils.hooks.activation_hooks.add_activation_sparsity_hook

```{autodoc2-docstring} src.utils.hooks.activation_hooks.add_activation_sparsity_hook
```
````

````{py:function} compute_sparsity_percentages(sparsity_stats: typing.Dict[str, typing.Dict[str, int]]) -> typing.Dict[str, float]
:canonical: src.utils.hooks.activation_hooks.compute_sparsity_percentages

```{autodoc2-docstring} src.utils.hooks.activation_hooks.compute_sparsity_percentages
```
````

````{py:function} print_activation_summary(statistics: typing.Dict[str, typing.Dict[str, float]], sparsity: typing.Optional[typing.Dict[str, float]] = None, dead_neurons: typing.Optional[typing.Dict[str, int]] = None) -> None
:canonical: src.utils.hooks.activation_hooks.print_activation_summary

```{autodoc2-docstring} src.utils.hooks.activation_hooks.print_activation_summary
```
````

````{py:function} remove_all_hooks(hook_data: typing.Dict[str, typing.Any]) -> None
:canonical: src.utils.hooks.activation_hooks.remove_all_hooks

```{autodoc2-docstring} src.utils.hooks.activation_hooks.remove_all_hooks
```
````
