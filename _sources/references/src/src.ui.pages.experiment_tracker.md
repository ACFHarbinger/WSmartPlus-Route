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

* - {py:obj}`_PALETTE <src.ui.pages.experiment_tracker._PALETTE>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker._PALETTE
    :summary:
    ```
* - {py:obj}`_STATUS_ICONS <src.ui.pages.experiment_tracker._STATUS_ICONS>`
  - ```{autodoc2-docstring} src.ui.pages.experiment_tracker._STATUS_ICONS
    :summary:
    ```
````

### API

````{py:data} _PALETTE
:canonical: src.ui.pages.experiment_tracker._PALETTE
:value: >
   ['#667eea', '#f093fb', '#4fd1c5', '#f6ad55', '#fc8181', '#90cdf4', '#9ae6b4', '#fbd38d', '#d6bcfa', ...

```{autodoc2-docstring} src.ui.pages.experiment_tracker._PALETTE
```

````

````{py:data} _STATUS_ICONS
:canonical: src.ui.pages.experiment_tracker._STATUS_ICONS
:value: >
   None

```{autodoc2-docstring} src.ui.pages.experiment_tracker._STATUS_ICONS
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
