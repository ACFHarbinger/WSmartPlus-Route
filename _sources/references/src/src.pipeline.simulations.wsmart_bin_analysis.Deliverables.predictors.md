# {py:mod}`src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors`

```{py:module} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Predictor <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor
    :summary:
    ```
````

### API

`````{py:class} Predictor(train_data: pandas.DataFrame, test_data: pandas.DataFrame, fit_name: str = None)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.__init__
```

````{py:method} get_pred_values(date) -> tuple[numpy.ndarray, numpy.ndarray]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.get_pred_values

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.get_pred_values
```

````

````{py:method} get_real_errors() -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.get_real_errors

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.get_real_errors
```

````

````{py:method} get_39mean_MSE() -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.get_39mean_MSE

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.get_39mean_MSE
```

````

````{py:method} get_MSE() -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.get_MSE

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.get_MSE
```

````

````{py:method} get_avg_dispersion() -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.get_avg_dispersion

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.get_avg_dispersion
```

````

````{py:method} get_pred_errors() -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.get_pred_errors

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.get_pred_errors
```

````

````{py:method} fit_39mean(train: pandas.DataFrame, test: pandas.DataFrame)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.fit_39mean

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.fit_39mean
```

````

````{py:method} fit_predict(train_data: pandas.DataFrame, test_data: pandas.DataFrame, fit_name: str = None)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.fit_predict

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.fit_predict
```

````

````{py:method} save_cache(train_data: pandas.DataFrame, test_data: pandas.DataFrame)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.save_cache

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.save_cache
```

````

````{py:method} load_cache()
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.load_cache

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.load_cache
```

````

````{py:method} deleate_cache()
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.deleate_cache

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.deleate_cache
```

````

````{py:method} plot_predictions(index, start_date: pandas.Timestamp, end_date: pandas.Timestamp, real_data: pandas.DataFrame, info: pandas.DataFrame, residuals_header: str, ylim=(-20, 60), fig_size: tuple = (9, 6))
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.plot_predictions

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.predictors.Predictor.plot_predictions
```

````

`````
