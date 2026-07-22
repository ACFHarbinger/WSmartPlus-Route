# {py:mod}`src.tracking.logging.modules.attention_emit`

```{py:module} src.tracking.logging.modules.attention_emit
```

```{autodoc2-docstring} src.tracking.logging.modules.attention_emit
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`send_attention_viz_to_gui <src.tracking.logging.modules.attention_emit.send_attention_viz_to_gui>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.attention_emit.send_attention_viz_to_gui
    :summary:
    ```
* - {py:obj}`maybe_emit_attention_viz <src.tracking.logging.modules.attention_emit.maybe_emit_attention_viz>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.attention_emit.maybe_emit_attention_viz
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ATTENTION_VIZ_MARKER <src.tracking.logging.modules.attention_emit.ATTENTION_VIZ_MARKER>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.attention_emit.ATTENTION_VIZ_MARKER
    :summary:
    ```
````

### API

````{py:data} ATTENTION_VIZ_MARKER
:canonical: src.tracking.logging.modules.attention_emit.ATTENTION_VIZ_MARKER
:value: >
   'ATTENTION_VIZ_START:'

```{autodoc2-docstring} src.tracking.logging.modules.attention_emit.ATTENTION_VIZ_MARKER
```

````

````{py:function} send_attention_viz_to_gui(snapshots: typing.List[typing.Dict[str, typing.Any]], phase: str, epoch: int, step: int, log_path: typing.Optional[str], lock: typing.Optional[threading.Lock] = None) -> None
:canonical: src.tracking.logging.modules.attention_emit.send_attention_viz_to_gui

```{autodoc2-docstring} src.tracking.logging.modules.attention_emit.send_attention_viz_to_gui
```
````

````{py:function} maybe_emit_attention_viz(model: typing.Any, cfg: typing.Any, phase: str = 'eval', epoch: int = 0, step: int = 0, log_path: typing.Optional[str] = None, lock: typing.Optional[threading.Lock] = None) -> bool
:canonical: src.tracking.logging.modules.attention_emit.maybe_emit_attention_viz

```{autodoc2-docstring} src.tracking.logging.modules.attention_emit.maybe_emit_attention_viz
```
````
