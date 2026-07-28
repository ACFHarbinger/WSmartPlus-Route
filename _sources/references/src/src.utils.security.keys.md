# {py:mod}`src.utils.security.keys`

```{py:module} src.utils.security.keys
```

```{autodoc2-docstring} src.utils.security.keys
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_set_param <src.utils.security.keys._set_param>`
  - ```{autodoc2-docstring} src.utils.security.keys._set_param
    :summary:
    ```
* - {py:obj}`generate_key <src.utils.security.keys.generate_key>`
  - ```{autodoc2-docstring} src.utils.security.keys.generate_key
    :summary:
    ```
* - {py:obj}`load_key <src.utils.security.keys.load_key>`
  - ```{autodoc2-docstring} src.utils.security.keys.load_key
    :summary:
    ```
````

### API

````{py:function} _set_param(config, param_name, param_value=None)
:canonical: src.utils.security.keys._set_param

```{autodoc2-docstring} src.utils.security.keys._set_param
```
````

````{py:function} generate_key(salt_size: int = 16, key_length: int = 32, hash_iterations: int = 100000, symkey_name: typing.Optional[str] = None, env_filename: str = '.env') -> typing.Tuple[bytes, bytes]
:canonical: src.utils.security.keys.generate_key

```{autodoc2-docstring} src.utils.security.keys.generate_key
```
````

````{py:function} load_key(symkey_name: typing.Optional[str] = None, env_filename: str = '.env') -> bytes
:canonical: src.utils.security.keys.load_key

```{autodoc2-docstring} src.utils.security.keys.load_key
```
````
