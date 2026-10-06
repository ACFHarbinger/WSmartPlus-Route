# {py:mod}`src.utils.security.directories`

```{py:module} src.utils.security.directories
```

```{autodoc2-docstring} src.utils.security.directories
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_validate_input_directory <src.utils.security.directories._validate_input_directory>`
  - ```{autodoc2-docstring} src.utils.security.directories._validate_input_directory
    :summary:
    ```
* - {py:obj}`encrypt_directory <src.utils.security.directories.encrypt_directory>`
  - ```{autodoc2-docstring} src.utils.security.directories.encrypt_directory
    :summary:
    ```
* - {py:obj}`decrypt_directory <src.utils.security.directories.decrypt_directory>`
  - ```{autodoc2-docstring} src.utils.security.directories.decrypt_directory
    :summary:
    ```
* - {py:obj}`encrypt_zip_directory <src.utils.security.directories.encrypt_zip_directory>`
  - ```{autodoc2-docstring} src.utils.security.directories.encrypt_zip_directory
    :summary:
    ```
* - {py:obj}`decrypt_zip <src.utils.security.directories.decrypt_zip>`
  - ```{autodoc2-docstring} src.utils.security.directories.decrypt_zip
    :summary:
    ```
````

### API

````{py:function} _validate_input_directory(input_dir: typing.Union[str, os.PathLike]) -> pathlib.Path
:canonical: src.utils.security.directories._validate_input_directory

```{autodoc2-docstring} src.utils.security.directories._validate_input_directory
```
````

````{py:function} encrypt_directory(key: bytes, input_dir: typing.Union[str, os.PathLike], output_dir: typing.Optional[typing.Union[str, os.PathLike]] = None) -> typing.List[bytes]
:canonical: src.utils.security.directories.encrypt_directory

```{autodoc2-docstring} src.utils.security.directories.encrypt_directory
```
````

````{py:function} decrypt_directory(key: bytes, input_dir: typing.Union[str, os.PathLike], output_dir: typing.Optional[typing.Union[str, os.PathLike]] = None) -> typing.List[typing.Union[str, bytes]]
:canonical: src.utils.security.directories.decrypt_directory

```{autodoc2-docstring} src.utils.security.directories.decrypt_directory
```
````

````{py:function} encrypt_zip_directory(key: bytes, input_dir: typing.Union[str, os.PathLike], output_enczip: typing.Optional[typing.Union[str, os.PathLike]] = None) -> bytes
:canonical: src.utils.security.directories.encrypt_zip_directory

```{autodoc2-docstring} src.utils.security.directories.encrypt_zip_directory
```
````

````{py:function} decrypt_zip(key: bytes, input_enczip: typing.Union[str, os.PathLike], output_dir: typing.Optional[typing.Union[str, os.PathLike]] = None) -> typing.Union[str, bytes]
:canonical: src.utils.security.directories.decrypt_zip

```{autodoc2-docstring} src.utils.security.directories.decrypt_zip
```
````
