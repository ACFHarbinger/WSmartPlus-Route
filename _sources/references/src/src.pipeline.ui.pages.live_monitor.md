# {py:mod}`src.pipeline.ui.pages.live_monitor`

```{py:module} src.pipeline.ui.pages.live_monitor
```

```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_read_raw_lines <src.pipeline.ui.pages.live_monitor._read_raw_lines>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor._read_raw_lines
    :summary:
    ```
* - {py:obj}`_render_log_viewer <src.pipeline.ui.pages.live_monitor._render_log_viewer>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor._render_log_viewer
    :summary:
    ```
* - {py:obj}`_render_live_kpis <src.pipeline.ui.pages.live_monitor._render_live_kpis>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor._render_live_kpis
    :summary:
    ```
* - {py:obj}`_render_live_chart <src.pipeline.ui.pages.live_monitor._render_live_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor._render_live_chart
    :summary:
    ```
* - {py:obj}`_render_sidebar_controls <src.pipeline.ui.pages.live_monitor._render_sidebar_controls>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor._render_sidebar_controls
    :summary:
    ```
* - {py:obj}`render_live_monitor <src.pipeline.ui.pages.live_monitor.render_live_monitor>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor.render_live_monitor
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_TARGET_METRICS <src.pipeline.ui.pages.live_monitor._TARGET_METRICS>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor._TARGET_METRICS
    :summary:
    ```
* - {py:obj}`_MAX_LOG_LINES <src.pipeline.ui.pages.live_monitor._MAX_LOG_LINES>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor._MAX_LOG_LINES
    :summary:
    ```
````

### API

````{py:data} _TARGET_METRICS
:canonical: src.pipeline.ui.pages.live_monitor._TARGET_METRICS
:value: >
   ['overflows', 'kg', 'ncol', 'km', 'kg/km', 'profit', 'cost', 'kg_lost']

```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor._TARGET_METRICS
```

````

````{py:data} _MAX_LOG_LINES
:canonical: src.pipeline.ui.pages.live_monitor._MAX_LOG_LINES
:value: >
   200

```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor._MAX_LOG_LINES
```

````

````{py:function} _read_raw_lines(log_path: str, max_lines: int = _MAX_LOG_LINES) -> typing.List[str]
:canonical: src.pipeline.ui.pages.live_monitor._read_raw_lines

```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor._read_raw_lines
```
````

````{py:function} _render_log_viewer(log_path: str) -> None
:canonical: src.pipeline.ui.pages.live_monitor._render_log_viewer

```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor._render_log_viewer
```
````

````{py:function} _render_live_kpis(entries: typing.List[logic.src.pipeline.ui.services.log_parser.DayLogEntry], policy: typing.Optional[str], sample_id: typing.Optional[int]) -> None
:canonical: src.pipeline.ui.pages.live_monitor._render_live_kpis

```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor._render_live_kpis
```
````

````{py:function} _render_live_chart(entries: typing.List[logic.src.pipeline.ui.services.log_parser.DayLogEntry], policy: typing.Optional[str], selected_metric: str) -> None
:canonical: src.pipeline.ui.pages.live_monitor._render_live_chart

```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor._render_live_chart
```
````

````{py:function} _render_sidebar_controls(policies: typing.List[str], samples: typing.List[int]) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.ui.pages.live_monitor._render_sidebar_controls

```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor._render_sidebar_controls
```
````

````{py:function} render_live_monitor() -> None
:canonical: src.pipeline.ui.pages.live_monitor.render_live_monitor

```{autodoc2-docstring} src.pipeline.ui.pages.live_monitor.render_live_monitor
```
````
