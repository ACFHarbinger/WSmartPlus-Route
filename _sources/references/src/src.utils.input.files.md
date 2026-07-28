# {py:mod}`src.utils.input.files`

```{py:module} src.utils.input.files
```

```{autodoc2-docstring} src.utils.input.files
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`read_json <src.utils.input.files.read_json>`
  - ```{autodoc2-docstring} src.utils.input.files.read_json
    :summary:
    ```
* - {py:obj}`zip_directory <src.utils.input.files.zip_directory>`
  - ```{autodoc2-docstring} src.utils.input.files.zip_directory
    :summary:
    ```
* - {py:obj}`extract_zip <src.utils.input.files.extract_zip>`
  - ```{autodoc2-docstring} src.utils.input.files.extract_zip
    :summary:
    ```
* - {py:obj}`confirm_proceed <src.utils.input.files.confirm_proceed>`
  - ```{autodoc2-docstring} src.utils.input.files.confirm_proceed
    :summary:
    ```
* - {py:obj}`compose_dirpath <src.utils.input.files.compose_dirpath>`
  - ```{autodoc2-docstring} src.utils.input.files.compose_dirpath
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`T <src.utils.input.files.T>`
  - ```{autodoc2-docstring} src.utils.input.files.T
    :summary:
    ```
````

### API

````{py:data} T
:canonical: src.utils.input.files.T
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} src.utils.input.files.T
```

````

````{py:function} read_json(json_path: str, lock: typing.Optional[threading.Lock] = None) -> typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Any]]
:canonical: src.utils.input.files.read_json

```{autodoc2-docstring} src.utils.input.files.read_json
```
````

````{py:function} zip_directory(input_dir: str, output_zip: str) -> None
:canonical: src.utils.input.files.zip_directory

```{autodoc2-docstring} src.utils.input.files.zip_directory
```
````

````{py:function} extract_zip(input_zip: str, output_dir: str) -> None
:canonical: src.utils.input.files.extract_zip

```{autodoc2-docstring} src.utils.input.files.extract_zip
```
````

````{py:function} confirm_proceed(default_no: bool = True, operation_name: str = 'update') -> bool
:canonical: src.utils.input.files.confirm_proceed

```{autodoc2-docstring} src.utils.input.files.confirm_proceed
```
````

````{py:function} compose_dirpath(fun: src.utils.input.files.T) -> src.utils.input.files.T
:canonical: src.utils.input.files.compose_dirpath

```{autodoc2-docstring} src.utils.input.files.compose_dirpath
```
````
