# {py:mod}`src.pipeline.features.eval.evaluate`

```{py:module} src.pipeline.features.eval.evaluate
```

```{autodoc2-docstring} src.pipeline.features.eval.evaluate
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_automatic_batch_size <src.pipeline.features.eval.evaluate.get_automatic_batch_size>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.evaluate.get_automatic_batch_size
    :summary:
    ```
* - {py:obj}`evaluate_policy <src.pipeline.features.eval.evaluate.evaluate_policy>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.evaluate.evaluate_policy
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`log <src.pipeline.features.eval.evaluate.log>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.evaluate.log
    :summary:
    ```
````

### API

````{py:data} log
:canonical: src.pipeline.features.eval.evaluate.log
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.features.eval.evaluate.log
```

````

````{py:function} get_automatic_batch_size(policy: typing.Any, env: typing.Any, data_loader: torch.utils.data.DataLoader, method: str = 'greedy', initial_batch_size: int = 1024, max_tries: int = 10, **kwargs) -> int
:canonical: src.pipeline.features.eval.evaluate.get_automatic_batch_size

```{autodoc2-docstring} src.pipeline.features.eval.evaluate.get_automatic_batch_size
```
````

````{py:function} evaluate_policy(policy: typing.Any, env: typing.Any, data_loader: torch.utils.data.DataLoader, method: str = 'greedy', return_results: bool = False, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.features.eval.evaluate.evaluate_policy

```{autodoc2-docstring} src.pipeline.features.eval.evaluate.evaluate_policy
```
````
