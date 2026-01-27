# {py:mod}`src.utils.security.crypto_utils`

```{py:module} src.utils.security.crypto_utils
```

```{autodoc2-docstring} src.utils.security.crypto_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_set_param <src.utils.security.crypto_utils._set_param>`
  - ```{autodoc2-docstring} src.utils.security.crypto_utils._set_param
    :summary:
    ```
* - {py:obj}`generate_key <src.utils.security.crypto_utils.generate_key>`
  - ```{autodoc2-docstring} src.utils.security.crypto_utils.generate_key
    :summary:
    ```
* - {py:obj}`load_key <src.utils.security.crypto_utils.load_key>`
  - ```{autodoc2-docstring} src.utils.security.crypto_utils.load_key
    :summary:
    ```
* - {py:obj}`encode_data <src.utils.security.crypto_utils.encode_data>`
  - ```{autodoc2-docstring} src.utils.security.crypto_utils.encode_data
    :summary:
    ```
* - {py:obj}`encrypt_file_data <src.utils.security.crypto_utils.encrypt_file_data>`
  - ```{autodoc2-docstring} src.utils.security.crypto_utils.encrypt_file_data
    :summary:
    ```
* - {py:obj}`decrypt_file_data <src.utils.security.crypto_utils.decrypt_file_data>`
  - ```{autodoc2-docstring} src.utils.security.crypto_utils.decrypt_file_data
    :summary:
    ```
* - {py:obj}`encrypt_directory <src.utils.security.crypto_utils.encrypt_directory>`
  - ```{autodoc2-docstring} src.utils.security.crypto_utils.encrypt_directory
    :summary:
    ```
* - {py:obj}`decrypt_directory <src.utils.security.crypto_utils.decrypt_directory>`
  - ```{autodoc2-docstring} src.utils.security.crypto_utils.decrypt_directory
    :summary:
    ```
* - {py:obj}`encrypt_zip_directory <src.utils.security.crypto_utils.encrypt_zip_directory>`
  - ```{autodoc2-docstring} src.utils.security.crypto_utils.encrypt_zip_directory
    :summary:
    ```
* - {py:obj}`decrypt_zip <src.utils.security.crypto_utils.decrypt_zip>`
  - ```{autodoc2-docstring} src.utils.security.crypto_utils.decrypt_zip
    :summary:
    ```
````

### API

````{py:function} _set_param(config, param_name, param_value=None)
:canonical: src.utils.security.crypto_utils._set_param

```{autodoc2-docstring} src.utils.security.crypto_utils._set_param
```
````

````{py:function} generate_key(salt_size: int = 16, key_length: int = 32, hash_iterations: int = 100000, symkey_name: typing.Optional[str] = None, env_filename: str = '.env') -> typing.Tuple[bytes, bytes]
:canonical: src.utils.security.crypto_utils.generate_key

```{autodoc2-docstring} src.utils.security.crypto_utils.generate_key
```
````

````{py:function} load_key(symkey_name: typing.Optional[str] = None, env_filename: str = '.env') -> bytes
:canonical: src.utils.security.crypto_utils.load_key

```{autodoc2-docstring} src.utils.security.crypto_utils.load_key
```
````

````{py:function} encode_data(data: typing.Any) -> bytes
:canonical: src.utils.security.crypto_utils.encode_data

```{autodoc2-docstring} src.utils.security.crypto_utils.encode_data
```
````

````{py:function} encrypt_file_data(key: bytes, input: typing.Union[str, os.PathLike, typing.Any], output_file: typing.Optional[typing.Union[str, os.PathLike]] = None) -> bytes
:canonical: src.utils.security.crypto_utils.encrypt_file_data

```{autodoc2-docstring} src.utils.security.crypto_utils.encrypt_file_data
```
````

````{py:function} decrypt_file_data(key: bytes, input: typing.Union[str, os.PathLike, typing.Any], output_file: typing.Optional[typing.Union[str, os.PathLike]] = None) -> str
:canonical: src.utils.security.crypto_utils.decrypt_file_data

```{autodoc2-docstring} src.utils.security.crypto_utils.decrypt_file_data
```
````

````{py:function} encrypt_directory(key: bytes, input_dir: typing.Union[str, os.PathLike], output_dir: typing.Optional[typing.Union[str, os.PathLike]] = None) -> typing.List[bytes]
:canonical: src.utils.security.crypto_utils.encrypt_directory

```{autodoc2-docstring} src.utils.security.crypto_utils.encrypt_directory
```
````

````{py:function} decrypt_directory(key: bytes, input_dir: typing.Union[str, os.PathLike], output_dir: typing.Optional[typing.Union[str, os.PathLike]] = None) -> typing.List[str]
:canonical: src.utils.security.crypto_utils.decrypt_directory

```{autodoc2-docstring} src.utils.security.crypto_utils.decrypt_directory
```
````

````{py:function} encrypt_zip_directory(key: bytes, input_dir: typing.Union[str, os.PathLike], output_enczip: typing.Optional[typing.Union[str, os.PathLike]] = None) -> bytes
:canonical: src.utils.security.crypto_utils.encrypt_zip_directory

```{autodoc2-docstring} src.utils.security.crypto_utils.encrypt_zip_directory
```
````

````{py:function} decrypt_zip(key: bytes, input_enczip: typing.Union[str, os.PathLike], output_dir: typing.Optional[typing.Union[str, os.PathLike]] = None) -> str
:canonical: src.utils.security.crypto_utils.decrypt_zip

```{autodoc2-docstring} src.utils.security.crypto_utils.decrypt_zip
```
````
