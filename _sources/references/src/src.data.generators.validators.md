# {py:mod}`src.data.generators.validators`

```{py:module} src.data.generators.validators
```

```{autodoc2-docstring} src.data.generators.validators
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_sanitize_area <src.data.generators.validators._sanitize_area>`
  - ```{autodoc2-docstring} src.data.generators.validators._sanitize_area
    :summary:
    ```
* - {py:obj}`_sanitize_waste <src.data.generators.validators._sanitize_waste>`
  - ```{autodoc2-docstring} src.data.generators.validators._sanitize_waste
    :summary:
    ```
* - {py:obj}`_get_graph_list <src.data.generators.validators._get_graph_list>`
  - ```{autodoc2-docstring} src.data.generators.validators._get_graph_list
    :summary:
    ```
* - {py:obj}`_validate_filename_args <src.data.generators.validators._validate_filename_args>`
  - ```{autodoc2-docstring} src.data.generators.validators._validate_filename_args
    :summary:
    ```
* - {py:obj}`_validate_problem_args <src.data.generators.validators._validate_problem_args>`
  - ```{autodoc2-docstring} src.data.generators.validators._validate_problem_args
    :summary:
    ```
* - {py:obj}`validate_gen_data_args <src.data.generators.validators.validate_gen_data_args>`
  - ```{autodoc2-docstring} src.data.generators.validators.validate_gen_data_args
    :summary:
    ```
````

### API

````{py:function} _sanitize_area(area: typing.Optional[str]) -> str
:canonical: src.data.generators.validators._sanitize_area

```{autodoc2-docstring} src.data.generators.validators._sanitize_area
```
````

````{py:function} _sanitize_waste(waste: typing.Optional[str]) -> str
:canonical: src.data.generators.validators._sanitize_waste

```{autodoc2-docstring} src.data.generators.validators._sanitize_waste
```
````

````{py:function} _get_graph_list(args: typing.Dict[str, typing.Any]) -> tuple[str, list[typing.Dict[str, typing.Any]]]
:canonical: src.data.generators.validators._get_graph_list

```{autodoc2-docstring} src.data.generators.validators._get_graph_list
```
````

````{py:function} _validate_filename_args(args: typing.Dict[str, typing.Any], dataset_count: int) -> None
:canonical: src.data.generators.validators._validate_filename_args

```{autodoc2-docstring} src.data.generators.validators._validate_filename_args
```
````

````{py:function} _validate_problem_args(args: typing.Dict[str, typing.Any]) -> None
:canonical: src.data.generators.validators._validate_problem_args

```{autodoc2-docstring} src.data.generators.validators._validate_problem_args
```
````

````{py:function} validate_gen_data_args(args: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]
:canonical: src.data.generators.validators.validate_gen_data_args

```{autodoc2-docstring} src.data.generators.validators.validate_gen_data_args
```
````
