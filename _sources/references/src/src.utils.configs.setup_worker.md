# {py:mod}`src.utils.configs.setup_worker`

```{py:module} src.utils.configs.setup_worker
```

```{autodoc2-docstring} src.utils.configs.setup_worker
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`setup_model <src.utils.configs.setup_worker.setup_model>`
  - ```{autodoc2-docstring} src.utils.configs.setup_worker.setup_model
    :summary:
    ```
````

### API

````{py:function} setup_model(policy: str, general_path: str, model_paths: typing.Dict[str, str], device: torch.device, lock: threading.Lock, temperature: float = 1.0, strategy: str = 'greedy') -> typing.Tuple[torch.nn.Module, typing.Dict[str, typing.Any]]
:canonical: src.utils.configs.setup_worker.setup_model

```{autodoc2-docstring} src.utils.configs.setup_worker.setup_model
```
````
