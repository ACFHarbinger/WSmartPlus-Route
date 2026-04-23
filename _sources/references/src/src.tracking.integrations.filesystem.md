# {py:mod}`src.tracking.integrations.filesystem`

```{py:module} src.tracking.integrations.filesystem
```

```{autodoc2-docstring} src.tracking.integrations.filesystem
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FilesystemTracker <src.tracking.integrations.filesystem.FilesystemTracker>`
  - ```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker
    :summary:
    ```
````

### API

`````{py:class} FilesystemTracker(run: typing.Optional[logic.src.tracking.core.run.Run])
:canonical: src.tracking.integrations.filesystem.FilesystemTracker

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker.__init__
```

````{py:method} on_load(path: str, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None) -> str
:canonical: src.tracking.integrations.filesystem.FilesystemTracker.on_load

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker.on_load
```

````

````{py:method} on_save(path: str, prev_hash: typing.Optional[str] = None, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None) -> str
:canonical: src.tracking.integrations.filesystem.FilesystemTracker.on_save

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker.on_save
```

````

````{py:method} on_stat(path: str) -> typing.Dict[str, typing.Any]
:canonical: src.tracking.integrations.filesystem.FilesystemTracker.on_stat

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker.on_stat
```

````

````{py:method} clear_cache() -> None
:canonical: src.tracking.integrations.filesystem.FilesystemTracker.clear_cache

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker.clear_cache
```

````

````{py:method} _hash_file(path: str) -> str
:canonical: src.tracking.integrations.filesystem.FilesystemTracker._hash_file

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker._hash_file
```

````

````{py:method} _get_stats(path: str) -> typing.Tuple[int, float]
:canonical: src.tracking.integrations.filesystem.FilesystemTracker._get_stats
:staticmethod:

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker._get_stats
```

````

`````
