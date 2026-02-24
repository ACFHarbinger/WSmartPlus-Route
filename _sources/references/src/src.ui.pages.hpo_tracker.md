# {py:mod}`src.ui.pages.hpo_tracker`

```{py:module} src.ui.pages.hpo_tracker
```

```{autodoc2-docstring} src.ui.pages.hpo_tracker
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`render_hpo_tracker <src.ui.pages.hpo_tracker.render_hpo_tracker>`
  - ```{autodoc2-docstring} src.ui.pages.hpo_tracker.render_hpo_tracker
    :summary:
    ```
* - {py:obj}`_render_hpo_kpis <src.ui.pages.hpo_tracker._render_hpo_kpis>`
  - ```{autodoc2-docstring} src.ui.pages.hpo_tracker._render_hpo_kpis
    :summary:
    ```
* - {py:obj}`_render_tab_parallel_coords <src.ui.pages.hpo_tracker._render_tab_parallel_coords>`
  - ```{autodoc2-docstring} src.ui.pages.hpo_tracker._render_tab_parallel_coords
    :summary:
    ```
* - {py:obj}`_render_tab_importance <src.ui.pages.hpo_tracker._render_tab_importance>`
  - ```{autodoc2-docstring} src.ui.pages.hpo_tracker._render_tab_importance
    :summary:
    ```
* - {py:obj}`_render_tab_history <src.ui.pages.hpo_tracker._render_tab_history>`
  - ```{autodoc2-docstring} src.ui.pages.hpo_tracker._render_tab_history
    :summary:
    ```
* - {py:obj}`_render_tab_contour <src.ui.pages.hpo_tracker._render_tab_contour>`
  - ```{autodoc2-docstring} src.ui.pages.hpo_tracker._render_tab_contour
    :summary:
    ```
* - {py:obj}`_load_study <src.ui.pages.hpo_tracker._load_study>`
  - ```{autodoc2-docstring} src.ui.pages.hpo_tracker._load_study
    :summary:
    ```
* - {py:obj}`_build_evaluator <src.ui.pages.hpo_tracker._build_evaluator>`
  - ```{autodoc2-docstring} src.ui.pages.hpo_tracker._build_evaluator
    :summary:
    ```
* - {py:obj}`_apply_layout <src.ui.pages.hpo_tracker._apply_layout>`
  - ```{autodoc2-docstring} src.ui.pages.hpo_tracker._apply_layout
    :summary:
    ```
* - {py:obj}`_render_trial_table <src.ui.pages.hpo_tracker._render_trial_table>`
  - ```{autodoc2-docstring} src.ui.pages.hpo_tracker._render_trial_table
    :summary:
    ```
````

### API

````{py:function} render_hpo_tracker(storage_url: str = 'sqlite:///assets/hpo/study.db', study_name: typing.Optional[str] = None, n_top_trials: int = 10, show_parallel_coords: bool = True, show_importance: bool = True, show_history: bool = True, show_contour: bool = False, importance_evaluator: str = 'fanova', height: int = 600) -> None
:canonical: src.ui.pages.hpo_tracker.render_hpo_tracker

```{autodoc2-docstring} src.ui.pages.hpo_tracker.render_hpo_tracker
```
````

````{py:function} _render_hpo_kpis(study: typing.Any, trials: typing.List[typing.Any], n_complete: int, optuna: typing.Any) -> None
:canonical: src.ui.pages.hpo_tracker._render_hpo_kpis

```{autodoc2-docstring} src.ui.pages.hpo_tracker._render_hpo_kpis
```
````

````{py:function} _render_tab_parallel_coords(study: typing.Any, ov: typing.Any, height: int, *args) -> None
:canonical: src.ui.pages.hpo_tracker._render_tab_parallel_coords

```{autodoc2-docstring} src.ui.pages.hpo_tracker._render_tab_parallel_coords
```
````

````{py:function} _render_tab_importance(study: typing.Any, ov: typing.Any, height: int, evaluator_name: str) -> None
:canonical: src.ui.pages.hpo_tracker._render_tab_importance

```{autodoc2-docstring} src.ui.pages.hpo_tracker._render_tab_importance
```
````

````{py:function} _render_tab_history(study: typing.Any, ov: typing.Any, height: int, *args) -> None
:canonical: src.ui.pages.hpo_tracker._render_tab_history

```{autodoc2-docstring} src.ui.pages.hpo_tracker._render_tab_history
```
````

````{py:function} _render_tab_contour(study: typing.Any, ov: typing.Any, height: int, *args) -> None
:canonical: src.ui.pages.hpo_tracker._render_tab_contour

```{autodoc2-docstring} src.ui.pages.hpo_tracker._render_tab_contour
```
````

````{py:function} _load_study(storage_url: str, study_name: typing.Optional[str], optuna: typing.Any) -> typing.Optional[typing.Any]
:canonical: src.ui.pages.hpo_tracker._load_study

```{autodoc2-docstring} src.ui.pages.hpo_tracker._load_study
```
````

````{py:function} _build_evaluator(name: str, optuna: typing.Any) -> typing.Any
:canonical: src.ui.pages.hpo_tracker._build_evaluator

```{autodoc2-docstring} src.ui.pages.hpo_tracker._build_evaluator
```
````

````{py:function} _apply_layout(fig: typing.Any, height: int) -> None
:canonical: src.ui.pages.hpo_tracker._apply_layout

```{autodoc2-docstring} src.ui.pages.hpo_tracker._apply_layout
```
````

````{py:function} _render_trial_table(completed_trials: typing.List[typing.Any], n_top: int) -> None
:canonical: src.ui.pages.hpo_tracker._render_trial_table

```{autodoc2-docstring} src.ui.pages.hpo_tracker._render_trial_table
```
````
