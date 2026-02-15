# {py:mod}`src.pipeline.ui.pages.training`

```{py:module} src.pipeline.ui.pages.training
```

```{autodoc2-docstring} src.pipeline.ui.pages.training
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_render_run_overview <src.pipeline.ui.pages.training._render_run_overview>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.training._render_run_overview
    :summary:
    ```
* - {py:obj}`_render_training_progress <src.pipeline.ui.pages.training._render_training_progress>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.training._render_training_progress
    :summary:
    ```
* - {py:obj}`_render_convergence_status <src.pipeline.ui.pages.training._render_convergence_status>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.training._render_convergence_status
    :summary:
    ```
* - {py:obj}`_render_lr_schedule <src.pipeline.ui.pages.training._render_lr_schedule>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.training._render_lr_schedule
    :summary:
    ```
* - {py:obj}`_render_training_kpis <src.pipeline.ui.pages.training._render_training_kpis>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.training._render_training_kpis
    :summary:
    ```
* - {py:obj}`_render_epoch_timing <src.pipeline.ui.pages.training._render_epoch_timing>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.training._render_epoch_timing
    :summary:
    ```
* - {py:obj}`_render_run_comparison <src.pipeline.ui.pages.training._render_run_comparison>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.training._render_run_comparison
    :summary:
    ```
* - {py:obj}`_render_all_metrics_table <src.pipeline.ui.pages.training._render_all_metrics_table>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.training._render_all_metrics_table
    :summary:
    ```
* - {py:obj}`render_training_monitor <src.pipeline.ui.pages.training.render_training_monitor>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.training.render_training_monitor
    :summary:
    ```
````

### API

````{py:function} _render_run_overview(selected_runs: typing.List[str]) -> None
:canonical: src.pipeline.ui.pages.training._render_run_overview

```{autodoc2-docstring} src.pipeline.ui.pages.training._render_run_overview
```
````

````{py:function} _render_training_progress(runs_data: typing.Dict[str, pandas.DataFrame], selected_runs: typing.List[str]) -> None
:canonical: src.pipeline.ui.pages.training._render_training_progress

```{autodoc2-docstring} src.pipeline.ui.pages.training._render_training_progress
```
````

````{py:function} _render_convergence_status(runs_data: typing.Dict[str, pandas.DataFrame]) -> None
:canonical: src.pipeline.ui.pages.training._render_convergence_status

```{autodoc2-docstring} src.pipeline.ui.pages.training._render_convergence_status
```
````

````{py:function} _render_lr_schedule(runs_data: typing.Dict[str, pandas.DataFrame]) -> None
:canonical: src.pipeline.ui.pages.training._render_lr_schedule

```{autodoc2-docstring} src.pipeline.ui.pages.training._render_lr_schedule
```
````

````{py:function} _render_training_kpis(runs_data: typing.Dict[str, pandas.DataFrame]) -> None
:canonical: src.pipeline.ui.pages.training._render_training_kpis

```{autodoc2-docstring} src.pipeline.ui.pages.training._render_training_kpis
```
````

````{py:function} _render_epoch_timing(runs_data: typing.Dict[str, pandas.DataFrame]) -> None
:canonical: src.pipeline.ui.pages.training._render_epoch_timing

```{autodoc2-docstring} src.pipeline.ui.pages.training._render_epoch_timing
```
````

````{py:function} _render_run_comparison(selected_runs: typing.List[str], runs_data: typing.Dict[str, pandas.DataFrame]) -> None
:canonical: src.pipeline.ui.pages.training._render_run_comparison

```{autodoc2-docstring} src.pipeline.ui.pages.training._render_run_comparison
```
````

````{py:function} _render_all_metrics_table(runs_data: typing.Dict[str, pandas.DataFrame]) -> None
:canonical: src.pipeline.ui.pages.training._render_all_metrics_table

```{autodoc2-docstring} src.pipeline.ui.pages.training._render_all_metrics_table
```
````

````{py:function} render_training_monitor() -> None
:canonical: src.pipeline.ui.pages.training.render_training_monitor

```{autodoc2-docstring} src.pipeline.ui.pages.training.render_training_monitor
```
````
