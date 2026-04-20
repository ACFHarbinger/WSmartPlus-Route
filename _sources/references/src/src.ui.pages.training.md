# {py:mod}`src.ui.pages.training`

```{py:module} src.ui.pages.training
```

```{autodoc2-docstring} src.ui.pages.training
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_render_run_overview <src.ui.pages.training._render_run_overview>`
  - ```{autodoc2-docstring} src.ui.pages.training._render_run_overview
    :summary:
    ```
* - {py:obj}`_render_training_progress <src.ui.pages.training._render_training_progress>`
  - ```{autodoc2-docstring} src.ui.pages.training._render_training_progress
    :summary:
    ```
* - {py:obj}`_render_convergence_status <src.ui.pages.training._render_convergence_status>`
  - ```{autodoc2-docstring} src.ui.pages.training._render_convergence_status
    :summary:
    ```
* - {py:obj}`render_training_monitor <src.ui.pages.training.render_training_monitor>`
  - ```{autodoc2-docstring} src.ui.pages.training.render_training_monitor
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`template_dir <src.ui.pages.training.template_dir>`
  - ```{autodoc2-docstring} src.ui.pages.training.template_dir
    :summary:
    ```
* - {py:obj}`jinja_env <src.ui.pages.training.jinja_env>`
  - ```{autodoc2-docstring} src.ui.pages.training.jinja_env
    :summary:
    ```
* - {py:obj}`STATUS_TEMPLATE <src.ui.pages.training.STATUS_TEMPLATE>`
  - ```{autodoc2-docstring} src.ui.pages.training.STATUS_TEMPLATE
    :summary:
    ```
````

### API

````{py:data} template_dir
:canonical: src.ui.pages.training.template_dir
:value: >
   'join(...)'

```{autodoc2-docstring} src.ui.pages.training.template_dir
```

````

````{py:data} jinja_env
:canonical: src.ui.pages.training.jinja_env
:value: >
   'Environment(...)'

```{autodoc2-docstring} src.ui.pages.training.jinja_env
```

````

````{py:data} STATUS_TEMPLATE
:canonical: src.ui.pages.training.STATUS_TEMPLATE
:value: >
   'get_template(...)'

```{autodoc2-docstring} src.ui.pages.training.STATUS_TEMPLATE
```

````

````{py:function} _render_run_overview(selected_runs: typing.List[str]) -> None
:canonical: src.ui.pages.training._render_run_overview

```{autodoc2-docstring} src.ui.pages.training._render_run_overview
```
````

````{py:function} _render_training_progress(runs_data: typing.Dict[str, pandas.DataFrame], selected_runs: typing.List[str]) -> None
:canonical: src.ui.pages.training._render_training_progress

```{autodoc2-docstring} src.ui.pages.training._render_training_progress
```
````

````{py:function} _render_convergence_status(runs_data: typing.Dict[str, pandas.DataFrame]) -> None
:canonical: src.ui.pages.training._render_convergence_status

```{autodoc2-docstring} src.ui.pages.training._render_convergence_status
```
````

````{py:function} render_training_monitor() -> None
:canonical: src.ui.pages.training.render_training_monitor

```{autodoc2-docstring} src.ui.pages.training.render_training_monitor
```
````
