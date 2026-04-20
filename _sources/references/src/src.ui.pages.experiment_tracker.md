# {py:mod}`src.ui.pages.experiment_tracker`

```{py:module} src.ui.pages.experiment_tracker
```

```{autodoc2-docstring} src.ui.pages.experiment_tracker
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_render_run_table <src.ui.pages.experiment_tracker._render_run_table>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker._render_run_table
    :summary:
    ```
* - {py:obj}`_render_run_detail <src.ui.pages.experiment_tracker._render_run_detail>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker._render_run_detail
    :summary:
    ```
* - {py:obj}`_render_params_table <src.ui.pages.experiment_tracker._render_params_table>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker._render_params_table
    :summary:
    ```
* - {py:obj}`_render_metric_explorer <src.ui.pages.experiment_tracker._render_metric_explorer>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker._render_metric_explorer
    :summary:
    ```
* - {py:obj}`_render_run_comparison <src.ui.pages.experiment_tracker._render_run_comparison>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker._render_run_comparison
    :summary:
    ```
* - {py:obj}`_render_artifacts <src.ui.pages.experiment_tracker._render_artifacts>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker._render_artifacts
    :summary:
    ```
* - {py:obj}`_render_dataset_events <src.ui.pages.experiment_tracker._render_dataset_events>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker._render_dataset_events
    :summary:
    ```
* - {py:obj}`_fmt_size <src.ui.pages.experiment_tracker._fmt_size>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker._fmt_size
    :summary:
    ```
* - {py:obj}`_render_tracker_sidebar <src.ui.pages.experiment_tracker._render_tracker_sidebar>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker._render_tracker_sidebar
    :summary:
    ```
* - {py:obj}`render_experiment_tracker <src.ui.pages.experiment_tracker.render_experiment_tracker>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker.render_experiment_tracker
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`template_dir <src.ui.pages.experiment_tracker.template_dir>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker.template_dir
    :summary:
    ```
* - {py:obj}`jinja_env <src.ui.pages.experiment_tracker.jinja_env>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker.jinja_env
    :summary:
    ```
* - {py:obj}`HOVER_TEMPLATE <src.ui.pages.experiment_tracker.HOVER_TEMPLATE>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker.HOVER_TEMPLATE
    :summary:
    ```
* - {py:obj}`_PALETTE <src.ui.pages.experiment_tracker._PALETTE>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker._PALETTE
    :summary:
    ```
* - {py:obj}`_EVENT_TYPE_ICONS <src.ui.pages.experiment_tracker._EVENT_TYPE_ICONS>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker._EVENT_TYPE_ICONS
    :summary:
    ```
````

### API

````{py:data} template_dir
:canonical: src.ui.pages.experiment_tracker.template_dir
:value: >
   'join(...)'

```{autodoc2-docstring} src.ui.pages.experiment_tracker.template_dir
```

````

````{py:data} jinja_env
:canonical: src.ui.pages.experiment_tracker.jinja_env
:value: >
   'Environment(...)'

```{autodoc2-docstring} src.ui.pages.experiment_tracker.jinja_env
```

````

````{py:data} HOVER_TEMPLATE
:canonical: src.ui.pages.experiment_tracker.HOVER_TEMPLATE
:value: >
   'get_template(...)'

```{autodoc2-docstring} src.ui.pages.experiment_tracker.HOVER_TEMPLATE
```

````

````{py:data} _PALETTE
:canonical: src.ui.pages.experiment_tracker._PALETTE
:value: >
   ['#667eea', '#f093fb', '#4fd1c5', '#f6ad55', '#fc8181', '#90cdf4', '#9ae6b4', '#fbd38d', '#d6bcfa', ...

```{autodoc2-docstring} src.ui.pages.experiment_tracker._PALETTE
```

````

````{py:function} _render_run_table(runs: typing.List[typing.Dict[str, typing.Any]], run_type_filter: typing.Optional[str]) -> typing.Optional[str]
:canonical: src.ui.pages.experiment_tracker._render_run_table

```{autodoc2-docstring} src.ui.pages.experiment_tracker._render_run_table
```
````

````{py:function} _render_run_detail(run_id: str) -> None
:canonical: src.ui.pages.experiment_tracker._render_run_detail

```{autodoc2-docstring} src.ui.pages.experiment_tracker._render_run_detail
```
````

````{py:function} _render_params_table(params: typing.Dict[str, typing.Any]) -> None
:canonical: src.ui.pages.experiment_tracker._render_params_table

```{autodoc2-docstring} src.ui.pages.experiment_tracker._render_params_table
```
````

````{py:function} _render_metric_explorer(run_id: str) -> None
:canonical: src.ui.pages.experiment_tracker._render_metric_explorer

```{autodoc2-docstring} src.ui.pages.experiment_tracker._render_metric_explorer
```
````

````{py:function} _render_run_comparison(runs: typing.List[typing.Dict[str, typing.Any]]) -> None
:canonical: src.ui.pages.experiment_tracker._render_run_comparison

```{autodoc2-docstring} src.ui.pages.experiment_tracker._render_run_comparison
```
````

````{py:function} _render_artifacts(run_id: str) -> None
:canonical: src.ui.pages.experiment_tracker._render_artifacts

```{autodoc2-docstring} src.ui.pages.experiment_tracker._render_artifacts
```
````

````{py:data} _EVENT_TYPE_ICONS
:canonical: src.ui.pages.experiment_tracker._EVENT_TYPE_ICONS
:value: >
   None

```{autodoc2-docstring} src.ui.pages.experiment_tracker._EVENT_TYPE_ICONS
```

````

````{py:function} _render_dataset_events(run_id: str) -> None
:canonical: src.ui.pages.experiment_tracker._render_dataset_events

```{autodoc2-docstring} src.ui.pages.experiment_tracker._render_dataset_events
```
````

````{py:function} _fmt_size(size_bytes: typing.Any) -> str
:canonical: src.ui.pages.experiment_tracker._fmt_size

```{autodoc2-docstring} src.ui.pages.experiment_tracker._fmt_size
```
````

````{py:function} _render_tracker_sidebar(run_types: typing.List[str]) -> typing.Dict[str, typing.Any]
:canonical: src.ui.pages.experiment_tracker._render_tracker_sidebar

```{autodoc2-docstring} src.ui.pages.experiment_tracker._render_tracker_sidebar
```
````

````{py:function} render_experiment_tracker() -> None
:canonical: src.ui.pages.experiment_tracker.render_experiment_tracker

```{autodoc2-docstring} src.ui.pages.experiment_tracker.render_experiment_tracker
```
````
