# {py:mod}`src.utils.model.processing`

```{py:module} src.utils.model.processing
```

```{autodoc2-docstring} src.utils.model.processing
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_inner_model <src.utils.model.processing.get_inner_model>`
  - ```{autodoc2-docstring} src.utils.model.processing.get_inner_model
    :summary:
    ```
* - {py:obj}`parse_softmax_temperature <src.utils.model.processing.parse_softmax_temperature>`
  - ```{autodoc2-docstring} src.utils.model.processing.parse_softmax_temperature
    :summary:
    ```
````

### API

````{py:function} get_inner_model(model: torch.nn.Module) -> torch.nn.Module
:canonical: src.utils.model.processing.get_inner_model

```{autodoc2-docstring} src.utils.model.processing.get_inner_model
```
````

````{py:function} parse_softmax_temperature(raw_temp: typing.Union[str, float]) -> float
:canonical: src.utils.model.processing.parse_softmax_temperature

```{autodoc2-docstring} src.utils.model.processing.parse_softmax_temperature
```
````
