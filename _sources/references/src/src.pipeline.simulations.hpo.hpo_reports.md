# {py:mod}`src.pipeline.simulations.hpo.hpo_reports`

```{py:module} src.pipeline.simulations.hpo.hpo_reports
```

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_reports
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_safe_study_slug <src.pipeline.simulations.hpo.hpo_reports._safe_study_slug>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_reports._safe_study_slug
    :summary:
    ```
* - {py:obj}`_write_plotly_figure <src.pipeline.simulations.hpo.hpo_reports._write_plotly_figure>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_reports._write_plotly_figure
    :summary:
    ```
* - {py:obj}`export_optuna_study_reports <src.pipeline.simulations.hpo.hpo_reports.export_optuna_study_reports>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_reports.export_optuna_study_reports
    :summary:
    ```
* - {py:obj}`export_optuna_study_from_storage <src.pipeline.simulations.hpo.hpo_reports.export_optuna_study_from_storage>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_reports.export_optuna_study_from_storage
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.simulations.hpo.hpo_reports.logger>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_reports.logger
    :summary:
    ```
* - {py:obj}`DEFAULT_REPORTS_SUBDIR <src.pipeline.simulations.hpo.hpo_reports.DEFAULT_REPORTS_SUBDIR>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_reports.DEFAULT_REPORTS_SUBDIR
    :summary:
    ```
* - {py:obj}`MIN_COMPLETED_FOR_PLOTS <src.pipeline.simulations.hpo.hpo_reports.MIN_COMPLETED_FOR_PLOTS>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_reports.MIN_COMPLETED_FOR_PLOTS
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.simulations.hpo.hpo_reports.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_reports.logger
```

````

````{py:data} DEFAULT_REPORTS_SUBDIR
:canonical: src.pipeline.simulations.hpo.hpo_reports.DEFAULT_REPORTS_SUBDIR
:value: >
   'join(...)'

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_reports.DEFAULT_REPORTS_SUBDIR
```

````

````{py:data} MIN_COMPLETED_FOR_PLOTS
:canonical: src.pipeline.simulations.hpo.hpo_reports.MIN_COMPLETED_FOR_PLOTS
:value: >
   2

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_reports.MIN_COMPLETED_FOR_PLOTS
```

````

````{py:function} _safe_study_slug(study_name: str) -> str
:canonical: src.pipeline.simulations.hpo.hpo_reports._safe_study_slug

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_reports._safe_study_slug
```
````

````{py:function} _write_plotly_figure(fig: typing.Any, stem: str, report_dir: str) -> typing.List[str]
:canonical: src.pipeline.simulations.hpo.hpo_reports._write_plotly_figure

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_reports._write_plotly_figure
```
````

````{py:function} export_optuna_study_reports(study: optuna.Study, output_dir: typing.Optional[str] = None, *, min_completed: int = MIN_COMPLETED_FOR_PLOTS) -> typing.Optional[str]
:canonical: src.pipeline.simulations.hpo.hpo_reports.export_optuna_study_reports

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_reports.export_optuna_study_reports
```
````

````{py:function} export_optuna_study_from_storage(storage_url: str, study_name: str, output_dir: typing.Optional[str] = None, *, min_completed: int = MIN_COMPLETED_FOR_PLOTS) -> typing.Optional[str]
:canonical: src.pipeline.simulations.hpo.hpo_reports.export_optuna_study_from_storage

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_reports.export_optuna_study_from_storage
```
````
