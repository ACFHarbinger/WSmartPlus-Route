# {py:mod}`src.utils.io.files`

```{py:module} src.utils.io.files
```

```{autodoc2-docstring} src.utils.io.files
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`read_json <src.utils.io.files.read_json>`
  - ```{autodoc2-docstring} src.utils.io.files.read_json
    :summary:
    ```
* - {py:obj}`zip_directory <src.utils.io.files.zip_directory>`
  - ```{autodoc2-docstring} src.utils.io.files.zip_directory
    :summary:
    ```
* - {py:obj}`extract_zip <src.utils.io.files.extract_zip>`
  - ```{autodoc2-docstring} src.utils.io.files.extract_zip
    :summary:
    ```
* - {py:obj}`confirm_proceed <src.utils.io.files.confirm_proceed>`
  - ```{autodoc2-docstring} src.utils.io.files.confirm_proceed
    :summary:
    ```
* - {py:obj}`compose_dirpath <src.utils.io.files.compose_dirpath>`
  - ```{autodoc2-docstring} src.utils.io.files.compose_dirpath
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`T <src.utils.io.files.T>`
  - ```{autodoc2-docstring} src.utils.io.files.T
    :summary:
    ```
````

### API

````{py:data} T
:canonical: src.utils.io.files.T
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} src.utils.io.files.T
```

````

````{py:function} read_json(json_path: str, lock: typing.Optional[threading.Lock] = None) -> typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Any]]
:canonical: src.utils.io.files.read_json

```{autodoc2-docstring} src.utils.io.files.read_json
```
````

````{py:function} zip_directory(input_dir: str, output_zip: str) -> None
:canonical: src.utils.io.files.zip_directory

```{autodoc2-docstring} src.utils.io.files.zip_directory
```
````

````{py:function} extract_zip(input_zip: str, output_dir: str) -> None
:canonical: src.utils.io.files.extract_zip

```{autodoc2-docstring} src.utils.io.files.extract_zip
```
````

````{py:function} confirm_proceed(default_no: bool = True, operation_name: str = 'update') -> bool
:canonical: src.utils.io.files.confirm_proceed

```{autodoc2-docstring} src.utils.io.files.confirm_proceed
```
````

````{py:function} compose_dirpath(fun: src.utils.io.files.T) -> src.utils.io.files.T
:canonical: src.utils.io.files.compose_dirpath

```{autodoc2-docstring} src.utils.io.files.compose_dirpath
```
````
