# {py:mod}`src.utils.security.data`

```{py:module} src.utils.security.data
```

```{autodoc2-docstring} src.utils.security.data
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`encode_data <src.utils.security.data.encode_data>`
  - ```{autodoc2-docstring} src.utils.security.data.encode_data
    :summary:
    ```
* - {py:obj}`encrypt_file_data <src.utils.security.data.encrypt_file_data>`
  - ```{autodoc2-docstring} src.utils.security.data.encrypt_file_data
    :summary:
    ```
* - {py:obj}`decrypt_file_data <src.utils.security.data.decrypt_file_data>`
  - ```{autodoc2-docstring} src.utils.security.data.decrypt_file_data
    :summary:
    ```
````

### API

````{py:function} encode_data(data: typing.Any) -> bytes
:canonical: src.utils.security.data.encode_data

```{autodoc2-docstring} src.utils.security.data.encode_data
```
````

````{py:function} encrypt_file_data(key: bytes, input: typing.Union[str, os.PathLike, typing.Any], output_file: typing.Optional[typing.Union[str, os.PathLike]] = None) -> bytes
:canonical: src.utils.security.data.encrypt_file_data

```{autodoc2-docstring} src.utils.security.data.encrypt_file_data
```
````

````{py:function} decrypt_file_data(key: bytes, input: typing.Union[str, os.PathLike, typing.Any], output_file: typing.Optional[typing.Union[str, os.PathLike]] = None) -> str
:canonical: src.utils.security.data.decrypt_file_data

```{autodoc2-docstring} src.utils.security.data.decrypt_file_data
```
````
