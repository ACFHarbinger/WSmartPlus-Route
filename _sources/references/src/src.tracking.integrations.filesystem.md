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

`````{py:class} FilesystemTracker(run: logic.src.tracking.core.run.Run)
:canonical: src.tracking.integrations.filesystem.FilesystemTracker

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker.__init__
```

````{py:method} track_load(file_path: str, num_samples: typing.Optional[int] = None, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None) -> None
:canonical: src.tracking.integrations.filesystem.FilesystemTracker.track_load

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker.track_load
```

````

````{py:method} track_generate(file_path: typing.Optional[str], num_samples: int, problem: str, graph_size: int, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None) -> None
:canonical: src.tracking.integrations.filesystem.FilesystemTracker.track_generate

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker.track_generate
```

````

````{py:method} track_mutation(description: str, num_samples: typing.Optional[int] = None, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None) -> None
:canonical: src.tracking.integrations.filesystem.FilesystemTracker.track_mutation

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker.track_mutation
```

````

````{py:method} track_save(file_path: str, num_samples: typing.Optional[int] = None, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None) -> None
:canonical: src.tracking.integrations.filesystem.FilesystemTracker.track_save

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker.track_save
```

````

````{py:method} watch(file_path: str) -> None
:canonical: src.tracking.integrations.filesystem.FilesystemTracker.watch

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker.watch
```

````

````{py:method} check_changes(paths: typing.Optional[typing.List[str]] = None) -> typing.List[str]
:canonical: src.tracking.integrations.filesystem.FilesystemTracker.check_changes

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker.check_changes
```

````

````{py:method} scan_directory(directory: str) -> None
:canonical: src.tracking.integrations.filesystem.FilesystemTracker.scan_directory

```{autodoc2-docstring} src.tracking.integrations.filesystem.FilesystemTracker.scan_directory
```

````

`````
