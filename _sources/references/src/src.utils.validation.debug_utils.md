# {py:mod}`src.utils.validation.debug_utils`

```{py:module} src.utils.validation.debug_utils
```

```{autodoc2-docstring} src.utils.validation.debug_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`watch <src.utils.validation.debug_utils.watch>`
  - ```{autodoc2-docstring} src.utils.validation.debug_utils.watch
    :summary:
    ```
* - {py:obj}`watch_all <src.utils.validation.debug_utils.watch_all>`
  - ```{autodoc2-docstring} src.utils.validation.debug_utils.watch_all
    :summary:
    ```
````

### API

````{py:function} watch(var_name: str, callback: typing.Optional[typing.Callable[[typing.Any, typing.Any, types.FrameType], None]] = None, *, frame_depth: int = 1) -> None
:canonical: src.utils.validation.debug_utils.watch

```{autodoc2-docstring} src.utils.validation.debug_utils.watch
```
````

````{py:function} watch_all(callback: typing.Optional[typing.Callable[[str, typing.Any, typing.Any, types.FrameType], None]] = None, *, frame_depth: int = 1) -> None
:canonical: src.utils.validation.debug_utils.watch_all

```{autodoc2-docstring} src.utils.validation.debug_utils.watch_all
```
````
