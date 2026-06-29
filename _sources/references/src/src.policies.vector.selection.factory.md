# {py:mod}`src.policies.vector.selection.factory`

```{py:module} src.policies.vector.selection.factory
```

```{autodoc2-docstring} src.policies.vector.selection.factory
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_selector_from_config <src.policies.vector.selection.factory.create_selector_from_config>`
  - ```{autodoc2-docstring} src.policies.vector.selection.factory.create_selector_from_config
    :summary:
    ```
* - {py:obj}`_get_strategy <src.policies.vector.selection.factory._get_strategy>`
  - ```{autodoc2-docstring} src.policies.vector.selection.factory._get_strategy
    :summary:
    ```
* - {py:obj}`_get_params <src.policies.vector.selection.factory._get_params>`
  - ```{autodoc2-docstring} src.policies.vector.selection.factory._get_params
    :summary:
    ```
* - {py:obj}`_create_combined_selector <src.policies.vector.selection.factory._create_combined_selector>`
  - ```{autodoc2-docstring} src.policies.vector.selection.factory._create_combined_selector
    :summary:
    ```
* - {py:obj}`_create_manager_selector <src.policies.vector.selection.factory._create_manager_selector>`
  - ```{autodoc2-docstring} src.policies.vector.selection.factory._create_manager_selector
    :summary:
    ```
* - {py:obj}`_get_strategy_params <src.policies.vector.selection.factory._get_strategy_params>`
  - ```{autodoc2-docstring} src.policies.vector.selection.factory._get_strategy_params
    :summary:
    ```
* - {py:obj}`get_vectorized_selector <src.policies.vector.selection.factory.get_vectorized_selector>`
  - ```{autodoc2-docstring} src.policies.vector.selection.factory.get_vectorized_selector
    :summary:
    ```
````

### API

````{py:function} create_selector_from_config(cfg: typing.Any) -> typing.Optional[src.policies.vector.selection.base.VectorizedSelector]
:canonical: src.policies.vector.selection.factory.create_selector_from_config

```{autodoc2-docstring} src.policies.vector.selection.factory.create_selector_from_config
```
````

````{py:function} _get_strategy(cfg: object) -> typing.Optional[str]
:canonical: src.policies.vector.selection.factory._get_strategy

```{autodoc2-docstring} src.policies.vector.selection.factory._get_strategy
```
````

````{py:function} _get_params(cfg: object) -> typing.Dict[str, typing.Any]
:canonical: src.policies.vector.selection.factory._get_params

```{autodoc2-docstring} src.policies.vector.selection.factory._get_params
```
````

````{py:function} _create_combined_selector(params: typing.Dict[str, typing.Any]) -> typing.Optional[src.policies.vector.selection.base.VectorizedSelector]
:canonical: src.policies.vector.selection.factory._create_combined_selector

```{autodoc2-docstring} src.policies.vector.selection.factory._create_combined_selector
```
````

````{py:function} _create_manager_selector(params: typing.Dict[str, typing.Any]) -> src.policies.vector.selection.base.VectorizedSelector
:canonical: src.policies.vector.selection.factory._create_manager_selector

```{autodoc2-docstring} src.policies.vector.selection.factory._create_manager_selector
```
````

````{py:function} _get_strategy_params(strategy: str, params: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]
:canonical: src.policies.vector.selection.factory._get_strategy_params

```{autodoc2-docstring} src.policies.vector.selection.factory._get_strategy_params
```
````

````{py:function} get_vectorized_selector(name: str, **kwargs: typing.Any) -> src.policies.vector.selection.base.VectorizedSelector
:canonical: src.policies.vector.selection.factory.get_vectorized_selector

```{autodoc2-docstring} src.policies.vector.selection.factory.get_vectorized_selector
```
````
