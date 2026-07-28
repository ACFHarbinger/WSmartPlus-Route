# {py:mod}`src.utils.model.checkpoint_utils`

```{py:module} src.utils.model.checkpoint_utils
```

```{autodoc2-docstring} src.utils.model.checkpoint_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`torch_load_cpu <src.utils.model.checkpoint_utils.torch_load_cpu>`
  - ```{autodoc2-docstring} src.utils.model.checkpoint_utils.torch_load_cpu
    :summary:
    ```
* - {py:obj}`load_data <src.utils.model.checkpoint_utils.load_data>`
  - ```{autodoc2-docstring} src.utils.model.checkpoint_utils.load_data
    :summary:
    ```
* - {py:obj}`_load_model_file <src.utils.model.checkpoint_utils._load_model_file>`
  - ```{autodoc2-docstring} src.utils.model.checkpoint_utils._load_model_file
    :summary:
    ```
````

### API

````{py:function} torch_load_cpu(load_path: str) -> typing.Any
:canonical: src.utils.model.checkpoint_utils.torch_load_cpu

```{autodoc2-docstring} src.utils.model.checkpoint_utils.torch_load_cpu
```
````

````{py:function} load_data(load_path: typing.Optional[str], resume: typing.Optional[str]) -> typing.Any
:canonical: src.utils.model.checkpoint_utils.load_data

```{autodoc2-docstring} src.utils.model.checkpoint_utils.load_data
```
````

````{py:function} _load_model_file(load_path: str, model: torch.nn.Module) -> typing.Tuple[torch.nn.Module, typing.Optional[typing.Dict[str, typing.Any]]]
:canonical: src.utils.model.checkpoint_utils._load_model_file

```{autodoc2-docstring} src.utils.model.checkpoint_utils._load_model_file
```
````
