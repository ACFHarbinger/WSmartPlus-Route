# {py:mod}`src.models.policies.selection.factory`

```{py:module} src.models.policies.selection.factory
```

```{autodoc2-docstring} src.models.policies.selection.factory
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_selector_from_config <src.models.policies.selection.factory.create_selector_from_config>`
  - ```{autodoc2-docstring} src.models.policies.selection.factory.create_selector_from_config
    :summary:
    ```
* - {py:obj}`_get_strategy <src.models.policies.selection.factory._get_strategy>`
  - ```{autodoc2-docstring} src.models.policies.selection.factory._get_strategy
    :summary:
    ```
* - {py:obj}`_get_params <src.models.policies.selection.factory._get_params>`
  - ```{autodoc2-docstring} src.models.policies.selection.factory._get_params
    :summary:
    ```
* - {py:obj}`_create_combined_selector <src.models.policies.selection.factory._create_combined_selector>`
  - ```{autodoc2-docstring} src.models.policies.selection.factory._create_combined_selector
    :summary:
    ```
* - {py:obj}`_create_manager_selector <src.models.policies.selection.factory._create_manager_selector>`
  - ```{autodoc2-docstring} src.models.policies.selection.factory._create_manager_selector
    :summary:
    ```
* - {py:obj}`_get_strategy_params <src.models.policies.selection.factory._get_strategy_params>`
  - ```{autodoc2-docstring} src.models.policies.selection.factory._get_strategy_params
    :summary:
    ```
* - {py:obj}`get_vectorized_selector <src.models.policies.selection.factory.get_vectorized_selector>`
  - ```{autodoc2-docstring} src.models.policies.selection.factory.get_vectorized_selector
    :summary:
    ```
````

### API

````{py:function} create_selector_from_config(cfg) -> typing.Optional[src.models.policies.selection.base.VectorizedSelector]
:canonical: src.models.policies.selection.factory.create_selector_from_config

```{autodoc2-docstring} src.models.policies.selection.factory.create_selector_from_config
```
````

````{py:function} _get_strategy(cfg: typing.Any) -> typing.Optional[str]
:canonical: src.models.policies.selection.factory._get_strategy

```{autodoc2-docstring} src.models.policies.selection.factory._get_strategy
```
````

````{py:function} _get_params(cfg: typing.Any) -> dict
:canonical: src.models.policies.selection.factory._get_params

```{autodoc2-docstring} src.models.policies.selection.factory._get_params
```
````

````{py:function} _create_combined_selector(params: dict) -> typing.Optional[src.models.policies.selection.base.VectorizedSelector]
:canonical: src.models.policies.selection.factory._create_combined_selector

```{autodoc2-docstring} src.models.policies.selection.factory._create_combined_selector
```
````

````{py:function} _create_manager_selector(params: dict) -> src.models.policies.selection.base.VectorizedSelector
:canonical: src.models.policies.selection.factory._create_manager_selector

```{autodoc2-docstring} src.models.policies.selection.factory._create_manager_selector
```
````

````{py:function} _get_strategy_params(strategy: str, params: dict) -> dict
:canonical: src.models.policies.selection.factory._get_strategy_params

```{autodoc2-docstring} src.models.policies.selection.factory._get_strategy_params
```
````

````{py:function} get_vectorized_selector(name: str, **kwargs) -> src.models.policies.selection.base.VectorizedSelector
:canonical: src.models.policies.selection.factory.get_vectorized_selector

```{autodoc2-docstring} src.models.policies.selection.factory.get_vectorized_selector
```
````
