# {py:mod}`src.constants.system`

```{py:module} src.constants.system
```

```{autodoc2-docstring} src.constants.system
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`update_lock_wait_time <src.constants.system.update_lock_wait_time>`
  - ```{autodoc2-docstring} src.constants.system.update_lock_wait_time
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CORE_LOCK_WAIT_TIME <src.constants.system.CORE_LOCK_WAIT_TIME>`
  - ```{autodoc2-docstring} src.constants.system.CORE_LOCK_WAIT_TIME
    :summary:
    ```
* - {py:obj}`LOCK_TIMEOUT <src.constants.system.LOCK_TIMEOUT>`
  - ```{autodoc2-docstring} src.constants.system.LOCK_TIMEOUT
    :summary:
    ```
* - {py:obj}`CONFIRM_TIMEOUT <src.constants.system.CONFIRM_TIMEOUT>`
  - ```{autodoc2-docstring} src.constants.system.CONFIRM_TIMEOUT
    :summary:
    ```
* - {py:obj}`FS_COMMANDS <src.constants.system.FS_COMMANDS>`
  - ```{autodoc2-docstring} src.constants.system.FS_COMMANDS
    :summary:
    ```
* - {py:obj}`OPERATION_MAP <src.constants.system.OPERATION_MAP>`
  - ```{autodoc2-docstring} src.constants.system.OPERATION_MAP
    :summary:
    ```
````

### API

````{py:data} CORE_LOCK_WAIT_TIME
:canonical: src.constants.system.CORE_LOCK_WAIT_TIME
:type: int
:value: >
   100

```{autodoc2-docstring} src.constants.system.CORE_LOCK_WAIT_TIME
```

````

````{py:data} LOCK_TIMEOUT
:canonical: src.constants.system.LOCK_TIMEOUT
:type: int
:value: >
   None

```{autodoc2-docstring} src.constants.system.LOCK_TIMEOUT
```

````

````{py:function} update_lock_wait_time(num_cpu_cores: typing.Optional[int] = None) -> int
:canonical: src.constants.system.update_lock_wait_time

```{autodoc2-docstring} src.constants.system.update_lock_wait_time
```
````

````{py:data} CONFIRM_TIMEOUT
:canonical: src.constants.system.CONFIRM_TIMEOUT
:type: int
:value: >
   30

```{autodoc2-docstring} src.constants.system.CONFIRM_TIMEOUT
```

````

````{py:data} FS_COMMANDS
:canonical: src.constants.system.FS_COMMANDS
:type: typing.List[str]
:value: >
   ['create', 'read', 'update', 'delete', 'cryptography']

```{autodoc2-docstring} src.constants.system.FS_COMMANDS
```

````

````{py:data} OPERATION_MAP
:canonical: src.constants.system.OPERATION_MAP
:type: typing.Dict[str, typing.Callable[[typing.Any, typing.Any], typing.Any]]
:value: >
   None

```{autodoc2-docstring} src.constants.system.OPERATION_MAP
```

````
