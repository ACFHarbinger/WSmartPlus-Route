# {py:mod}`src.tracking.logging.modules.policy_viz_emit`

```{py:module} src.tracking.logging.modules.policy_viz_emit
```

```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolicyVizStreamSession <src.tracking.logging.modules.policy_viz_emit.PolicyVizStreamSession>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit.PolicyVizStreamSession
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`detect_policy_viz_type <src.tracking.logging.modules.policy_viz_emit.detect_policy_viz_type>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit.detect_policy_viz_type
    :summary:
    ```
* - {py:obj}`_collect_viz_sources <src.tracking.logging.modules.policy_viz_emit._collect_viz_sources>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit._collect_viz_sources
    :summary:
    ```
* - {py:obj}`maybe_emit_policy_viz <src.tracking.logging.modules.policy_viz_emit.maybe_emit_policy_viz>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit.maybe_emit_policy_viz
    :summary:
    ```
* - {py:obj}`send_policy_viz_to_gui <src.tracking.logging.modules.policy_viz_emit.send_policy_viz_to_gui>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit.send_policy_viz_to_gui
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`POLICY_VIZ_MARKER <src.tracking.logging.modules.policy_viz_emit.POLICY_VIZ_MARKER>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit.POLICY_VIZ_MARKER
    :summary:
    ```
* - {py:obj}`STREAM_INTERVAL_SEC <src.tracking.logging.modules.policy_viz_emit.STREAM_INTERVAL_SEC>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit.STREAM_INTERVAL_SEC
    :summary:
    ```
````

### API

````{py:data} POLICY_VIZ_MARKER
:canonical: src.tracking.logging.modules.policy_viz_emit.POLICY_VIZ_MARKER
:value: >
   'POLICY_VIZ_START:'

```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit.POLICY_VIZ_MARKER
```

````

````{py:data} STREAM_INTERVAL_SEC
:canonical: src.tracking.logging.modules.policy_viz_emit.STREAM_INTERVAL_SEC
:value: >
   0.5

```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit.STREAM_INTERVAL_SEC
```

````

````{py:function} detect_policy_viz_type(viz_data: typing.Dict[str, typing.List[typing.Any]]) -> str
:canonical: src.tracking.logging.modules.policy_viz_emit.detect_policy_viz_type

```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit.detect_policy_viz_type
```
````

````{py:function} _collect_viz_sources(source: typing.Any) -> typing.List[typing.Any]
:canonical: src.tracking.logging.modules.policy_viz_emit._collect_viz_sources

```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit._collect_viz_sources
```
````

`````{py:class} PolicyVizStreamSession(source: typing.Any, policy: str, sample_idx: int, day: int, log_path: typing.Optional[str], lock: typing.Optional[threading.Lock] = None, interval_sec: float = STREAM_INTERVAL_SEC)
:canonical: src.tracking.logging.modules.policy_viz_emit.PolicyVizStreamSession

```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit.PolicyVizStreamSession
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit.PolicyVizStreamSession.__init__
```

````{py:method} __enter__() -> src.tracking.logging.modules.policy_viz_emit.PolicyVizStreamSession
:canonical: src.tracking.logging.modules.policy_viz_emit.PolicyVizStreamSession.__enter__

```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit.PolicyVizStreamSession.__enter__
```

````

````{py:method} __exit__(exc_type: typing.Any, exc_val: typing.Any, exc_tb: typing.Any) -> None
:canonical: src.tracking.logging.modules.policy_viz_emit.PolicyVizStreamSession.__exit__

```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit.PolicyVizStreamSession.__exit__
```

````

````{py:method} _emit_loop() -> None
:canonical: src.tracking.logging.modules.policy_viz_emit.PolicyVizStreamSession._emit_loop

```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit.PolicyVizStreamSession._emit_loop
```

````

`````

````{py:function} maybe_emit_policy_viz(source: typing.Any, policy: str, sample_idx: int, day: int, log_path: typing.Optional[str], lock: typing.Optional[threading.Lock] = None) -> bool
:canonical: src.tracking.logging.modules.policy_viz_emit.maybe_emit_policy_viz

```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit.maybe_emit_policy_viz
```
````

````{py:function} send_policy_viz_to_gui(viz_data: typing.Dict[str, typing.List[typing.Any]], policy: str, sample_idx: int, day: int, log_path: typing.Optional[str], lock: typing.Optional[threading.Lock] = None) -> None
:canonical: src.tracking.logging.modules.policy_viz_emit.send_policy_viz_to_gui

```{autodoc2-docstring} src.tracking.logging.modules.policy_viz_emit.send_policy_viz_to_gui
```
````
