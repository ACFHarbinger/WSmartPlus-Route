# {py:mod}`src.tracking.logging.visualization.heatmaps`

```{py:module} src.tracking.logging.visualization.heatmaps
```

```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_resolve_policy_module <src.tracking.logging.visualization.heatmaps._resolve_policy_module>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps._resolve_policy_module
    :summary:
    ```
* - {py:obj}`_resolve_encoder <src.tracking.logging.visualization.heatmaps._resolve_encoder>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps._resolve_encoder
    :summary:
    ```
* - {py:obj}`extract_attention_matrix <src.tracking.logging.visualization.heatmaps.extract_attention_matrix>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps.extract_attention_matrix
    :summary:
    ```
* - {py:obj}`render_attention_heatmap_png <src.tracking.logging.visualization.heatmaps.render_attention_heatmap_png>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps.render_attention_heatmap_png
    :summary:
    ```
* - {py:obj}`capture_runtime_attention <src.tracking.logging.visualization.heatmaps.capture_runtime_attention>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps.capture_runtime_attention
    :summary:
    ```
* - {py:obj}`plot_attention_heatmaps <src.tracking.logging.visualization.heatmaps.plot_attention_heatmaps>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps.plot_attention_heatmaps
    :summary:
    ```
* - {py:obj}`log_attention_heatmaps_to_backends <src.tracking.logging.visualization.heatmaps.log_attention_heatmaps_to_backends>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps.log_attention_heatmaps_to_backends
    :summary:
    ```
* - {py:obj}`maybe_log_eval_attention_heatmaps <src.tracking.logging.visualization.heatmaps.maybe_log_eval_attention_heatmaps>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps.maybe_log_eval_attention_heatmaps
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.tracking.logging.visualization.heatmaps.logger>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.tracking.logging.visualization.heatmaps.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps.logger
```

````

````{py:function} _resolve_policy_module(model: typing.Any) -> typing.Any
:canonical: src.tracking.logging.visualization.heatmaps._resolve_policy_module

```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps._resolve_policy_module
```
````

````{py:function} _resolve_encoder(policy: typing.Any) -> typing.Optional[torch.nn.Module]
:canonical: src.tracking.logging.visualization.heatmaps._resolve_encoder

```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps._resolve_encoder
```
````

````{py:function} extract_attention_matrix(tensor: torch.Tensor, head_idx: int = 0, batch_idx: int = 0) -> typing.Optional[numpy.ndarray]
:canonical: src.tracking.logging.visualization.heatmaps.extract_attention_matrix

```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps.extract_attention_matrix
```
````

````{py:function} render_attention_heatmap_png(matrix: numpy.ndarray, title: str, output_path: str) -> str
:canonical: src.tracking.logging.visualization.heatmaps.render_attention_heatmap_png

```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps.render_attention_heatmap_png
```
````

````{py:function} capture_runtime_attention(model: typing.Any, batch: typing.Any, device: typing.Optional[torch.device] = None, head_idx: int = 0) -> typing.List[typing.Tuple[int, numpy.ndarray]]
:canonical: src.tracking.logging.visualization.heatmaps.capture_runtime_attention

```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps.capture_runtime_attention
```
````

````{py:function} plot_attention_heatmaps(model: typing.Any, output_dir: str, epoch: int = 0) -> typing.List[str]
:canonical: src.tracking.logging.visualization.heatmaps.plot_attention_heatmaps

```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps.plot_attention_heatmaps
```
````

````{py:function} log_attention_heatmaps_to_backends(image_paths: typing.Dict[str, str], step: int = 0, wandb_mode: str = 'disabled', tb_writer: typing.Any = None) -> None
:canonical: src.tracking.logging.visualization.heatmaps.log_attention_heatmaps_to_backends

```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps.log_attention_heatmaps_to_backends
```
````

````{py:function} maybe_log_eval_attention_heatmaps(model: typing.Any, batch: typing.Any, cfg: typing.Any, output_subdir: str = 'eval_attention', step: int = 0, tb_writer: typing.Any = None, phase: str = 'eval', epoch: int = 0) -> typing.List[str]
:canonical: src.tracking.logging.visualization.heatmaps.maybe_log_eval_attention_heatmaps

```{autodoc2-docstring} src.tracking.logging.visualization.heatmaps.maybe_log_eval_attention_heatmaps
```
````
