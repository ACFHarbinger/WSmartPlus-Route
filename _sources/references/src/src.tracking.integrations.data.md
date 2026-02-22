# {py:mod}`src.tracking.integrations.data`

```{py:module} src.tracking.integrations.data
```

```{autodoc2-docstring} src.tracking.integrations.data
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RuntimeDataTracker <src.tracking.integrations.data.RuntimeDataTracker>`
  - ```{autodoc2-docstring} src.tracking.integrations.data.RuntimeDataTracker
    :summary:
    ```
````

### API

`````{py:class} RuntimeDataTracker(run: typing.Any, max_fields: int = 32)
:canonical: src.tracking.integrations.data.RuntimeDataTracker

```{autodoc2-docstring} src.tracking.integrations.data.RuntimeDataTracker
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.integrations.data.RuntimeDataTracker.__init__
```

````{py:method} snapshot(data: typing.Any, tag: str, step: typing.Optional[int] = None) -> None
:canonical: src.tracking.integrations.data.RuntimeDataTracker.snapshot

```{autodoc2-docstring} src.tracking.integrations.data.RuntimeDataTracker.snapshot
```

````

````{py:method} on_load(data: typing.Any, num_samples: typing.Optional[int] = None, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None) -> None
:canonical: src.tracking.integrations.data.RuntimeDataTracker.on_load

```{autodoc2-docstring} src.tracking.integrations.data.RuntimeDataTracker.on_load
```

````

````{py:method} on_regenerate(data: typing.Any, epoch: int, num_samples: typing.Optional[int] = None, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None) -> None
:canonical: src.tracking.integrations.data.RuntimeDataTracker.on_regenerate

```{autodoc2-docstring} src.tracking.integrations.data.RuntimeDataTracker.on_regenerate
```

````

````{py:method} on_augment(data: typing.Any, description: str, step: typing.Optional[int] = None, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None) -> None
:canonical: src.tracking.integrations.data.RuntimeDataTracker.on_augment

```{autodoc2-docstring} src.tracking.integrations.data.RuntimeDataTracker.on_augment
```

````

````{py:method} field_drift(field: str, stat: str = 'mean', window: int = 2) -> typing.Optional[float]
:canonical: src.tracking.integrations.data.RuntimeDataTracker.field_drift

```{autodoc2-docstring} src.tracking.integrations.data.RuntimeDataTracker.field_drift
```

````

````{py:method} _extract_fields(data: typing.Any) -> typing.Dict[str, torch.Tensor]
:canonical: src.tracking.integrations.data.RuntimeDataTracker._extract_fields

```{autodoc2-docstring} src.tracking.integrations.data.RuntimeDataTracker._extract_fields
```

````

````{py:method} _compute_stats(fields: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, typing.Dict[str, typing.Any]]
:canonical: src.tracking.integrations.data.RuntimeDataTracker._compute_stats
:staticmethod:

```{autodoc2-docstring} src.tracking.integrations.data.RuntimeDataTracker._compute_stats
```

````

````{py:method} _log_stats(stats: typing.Dict[str, typing.Dict[str, typing.Any]], tag: str, step: typing.Optional[int]) -> None
:canonical: src.tracking.integrations.data.RuntimeDataTracker._log_stats

```{autodoc2-docstring} src.tracking.integrations.data.RuntimeDataTracker._log_stats
```

````

````{py:method} _infer_size(data: typing.Any) -> typing.Optional[int]
:canonical: src.tracking.integrations.data.RuntimeDataTracker._infer_size
:staticmethod:

```{autodoc2-docstring} src.tracking.integrations.data.RuntimeDataTracker._infer_size
```

````

`````
