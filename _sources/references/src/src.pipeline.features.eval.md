# {py:mod}`src.pipeline.features.eval`

```{py:module} src.pipeline.features.eval
```

```{autodoc2-docstring} src.pipeline.features.eval
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

src.pipeline.features.eval.evaluators
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.pipeline.features.eval.zenml_eval_pipeline
src.pipeline.features.eval.evaluate
src.pipeline.features.eval.eval_base
src.pipeline.features.eval.validation
src.pipeline.features.eval.engine
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_evaluate_model <src.pipeline.features.eval.run_evaluate_model>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.run_evaluate_model
    :summary:
    ```
* - {py:obj}`_run_eval_via_zenml <src.pipeline.features.eval._run_eval_via_zenml>`
  - ```{autodoc2-docstring} src.pipeline.features.eval._run_eval_via_zenml
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.features.eval.logger>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.logger
    :summary:
    ```
* - {py:obj}`__all__ <src.pipeline.features.eval.__all__>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.__all__
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.features.eval.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.features.eval.logger
```

````

````{py:function} run_evaluate_model(cfg: logic.src.configs.Config, sinks: typing.Optional[typing.List[typing.Any]] = None) -> None
:canonical: src.pipeline.features.eval.run_evaluate_model

```{autodoc2-docstring} src.pipeline.features.eval.run_evaluate_model
```
````

````{py:function} _run_eval_via_zenml(cfg: logic.src.configs.Config) -> None
:canonical: src.pipeline.features.eval._run_eval_via_zenml

```{autodoc2-docstring} src.pipeline.features.eval._run_eval_via_zenml
```
````

````{py:data} __all__
:canonical: src.pipeline.features.eval.__all__
:value: >
   ['run_evaluate_model', 'eval_dataset', 'eval_dataset_mp', 'get_best', 'validate_eval_args']

```{autodoc2-docstring} src.pipeline.features.eval.__all__
```

````
