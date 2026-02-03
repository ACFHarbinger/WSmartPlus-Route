# {py:mod}`src.pipeline.ui.components.sidebar`

```{py:module} src.pipeline.ui.components.sidebar
```

```{autodoc2-docstring} src.pipeline.ui.components.sidebar
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`render_mode_selector <src.pipeline.ui.components.sidebar.render_mode_selector>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.sidebar.render_mode_selector
    :summary:
    ```
* - {py:obj}`render_auto_refresh_toggle <src.pipeline.ui.components.sidebar.render_auto_refresh_toggle>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.sidebar.render_auto_refresh_toggle
    :summary:
    ```
* - {py:obj}`render_training_controls <src.pipeline.ui.components.sidebar.render_training_controls>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.sidebar.render_training_controls
    :summary:
    ```
* - {py:obj}`render_simulation_controls <src.pipeline.ui.components.sidebar.render_simulation_controls>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.sidebar.render_simulation_controls
    :summary:
    ```
* - {py:obj}`render_about_section <src.pipeline.ui.components.sidebar.render_about_section>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.sidebar.render_about_section
    :summary:
    ```
````

### API

````{py:function} render_mode_selector() -> str
:canonical: src.pipeline.ui.components.sidebar.render_mode_selector

```{autodoc2-docstring} src.pipeline.ui.components.sidebar.render_mode_selector
```
````

````{py:function} render_auto_refresh_toggle() -> typing.Tuple[bool, int]
:canonical: src.pipeline.ui.components.sidebar.render_auto_refresh_toggle

```{autodoc2-docstring} src.pipeline.ui.components.sidebar.render_auto_refresh_toggle
```
````

````{py:function} render_training_controls(available_runs: typing.List[str], available_metrics: typing.List[str]) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.ui.components.sidebar.render_training_controls

```{autodoc2-docstring} src.pipeline.ui.components.sidebar.render_training_controls
```
````

````{py:function} render_simulation_controls(available_logs: typing.List[str], policies: typing.List[str], samples: typing.List[int], day_range: typing.Tuple[int, int]) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.ui.components.sidebar.render_simulation_controls

```{autodoc2-docstring} src.pipeline.ui.components.sidebar.render_simulation_controls
```
````

````{py:function} render_about_section() -> None
:canonical: src.pipeline.ui.components.sidebar.render_about_section

```{autodoc2-docstring} src.pipeline.ui.components.sidebar.render_about_section
```
````
