# {py:mod}`src.pipeline.features.eval.drift_detection`

```{py:module} src.pipeline.features.eval.drift_detection
```

```{autodoc2-docstring} src.pipeline.features.eval.drift_detection
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_drift_detection <src.pipeline.features.eval.drift_detection.run_drift_detection>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.drift_detection.run_drift_detection
    :summary:
    ```
* - {py:obj}`run_column_drift_suite <src.pipeline.features.eval.drift_detection.run_column_drift_suite>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.drift_detection.run_column_drift_suite
    :summary:
    ```
* - {py:obj}`load_and_flatten <src.pipeline.features.eval.drift_detection.load_and_flatten>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.drift_detection.load_and_flatten
    :summary:
    ```
* - {py:obj}`_npz_to_dataframe <src.pipeline.features.eval.drift_detection._npz_to_dataframe>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.drift_detection._npz_to_dataframe
    :summary:
    ```
* - {py:obj}`_check_evidently <src.pipeline.features.eval.drift_detection._check_evidently>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.drift_detection._check_evidently
    :summary:
    ```
* - {py:obj}`_log_drift_summary <src.pipeline.features.eval.drift_detection._log_drift_summary>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.drift_detection._log_drift_summary
    :summary:
    ```
* - {py:obj}`_build_arg_parser <src.pipeline.features.eval.drift_detection._build_arg_parser>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.drift_detection._build_arg_parser
    :summary:
    ```
* - {py:obj}`main <src.pipeline.features.eval.drift_detection.main>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.drift_detection.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`log <src.pipeline.features.eval.drift_detection.log>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.drift_detection.log
    :summary:
    ```
````

### API

````{py:data} log
:canonical: src.pipeline.features.eval.drift_detection.log
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.pipeline.features.eval.drift_detection.log
```

````

````{py:function} run_drift_detection(reference_path: str, current_path: str, output_dir: str = 'assets/drift_reports', report_filename: typing.Optional[str] = None, feature_columns: typing.Optional[typing.List[str]] = None, target_column: typing.Optional[str] = None, problem: str = 'vrpp', stattest: str = 'ks', stattest_threshold: float = 0.05) -> str
:canonical: src.pipeline.features.eval.drift_detection.run_drift_detection

```{autodoc2-docstring} src.pipeline.features.eval.drift_detection.run_drift_detection
```
````

````{py:function} run_column_drift_suite(reference_path: str, current_path: str, columns: typing.Optional[typing.List[str]] = None, problem: str = 'vrpp', output_dir: str = 'assets/drift_reports', stattest: str = 'ks', max_columns: int = 10) -> str
:canonical: src.pipeline.features.eval.drift_detection.run_column_drift_suite

```{autodoc2-docstring} src.pipeline.features.eval.drift_detection.run_column_drift_suite
```
````

````{py:function} load_and_flatten(path: str, problem: str = 'vrpp') -> pandas.DataFrame
:canonical: src.pipeline.features.eval.drift_detection.load_and_flatten

```{autodoc2-docstring} src.pipeline.features.eval.drift_detection.load_and_flatten
```
````

````{py:function} _npz_to_dataframe(path: str) -> pandas.DataFrame
:canonical: src.pipeline.features.eval.drift_detection._npz_to_dataframe

```{autodoc2-docstring} src.pipeline.features.eval.drift_detection._npz_to_dataframe
```
````

````{py:function} _check_evidently() -> None
:canonical: src.pipeline.features.eval.drift_detection._check_evidently

```{autodoc2-docstring} src.pipeline.features.eval.drift_detection._check_evidently
```
````

````{py:function} _log_drift_summary(report: typing.Any) -> None
:canonical: src.pipeline.features.eval.drift_detection._log_drift_summary

```{autodoc2-docstring} src.pipeline.features.eval.drift_detection._log_drift_summary
```
````

````{py:function} _build_arg_parser() -> argparse.ArgumentParser
:canonical: src.pipeline.features.eval.drift_detection._build_arg_parser

```{autodoc2-docstring} src.pipeline.features.eval.drift_detection._build_arg_parser
```
````

````{py:function} main() -> None
:canonical: src.pipeline.features.eval.drift_detection.main

```{autodoc2-docstring} src.pipeline.features.eval.drift_detection.main
```
````
