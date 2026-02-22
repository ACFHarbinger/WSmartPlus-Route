# {py:mod}`src.utils.functions.parallel`

```{py:module} src.utils.functions.parallel
```

```{autodoc2-docstring} src.utils.functions.parallel
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_all_in_pool <src.utils.functions.parallel.run_all_in_pool>`
  - ```{autodoc2-docstring} src.utils.functions.parallel.run_all_in_pool
    :summary:
    ```
````

### API

````{py:function} run_all_in_pool(func: typing.Callable[..., typing.Any], directory: str, dataset: typing.List[typing.Any], *, cpus: typing.Optional[int] = None, offset: int = 0, n: typing.Optional[int] = None, progress_bar_mininterval: float = 0.1, use_multiprocessing: bool = True) -> typing.Tuple[typing.List[typing.Any], int]
:canonical: src.utils.functions.parallel.run_all_in_pool

```{autodoc2-docstring} src.utils.functions.parallel.run_all_in_pool
```
````
