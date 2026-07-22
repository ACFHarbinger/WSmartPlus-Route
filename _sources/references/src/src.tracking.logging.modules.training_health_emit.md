# {py:mod}`src.tracking.logging.modules.training_health_emit`

```{py:module} src.tracking.logging.modules.training_health_emit
```

```{autodoc2-docstring} src.tracking.logging.modules.training_health_emit
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`emit_training_health_alert <src.tracking.logging.modules.training_health_emit.emit_training_health_alert>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.training_health_emit.emit_training_health_alert
    :summary:
    ```
* - {py:obj}`_default_message <src.tracking.logging.modules.training_health_emit._default_message>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.training_health_emit._default_message
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TRAINING_HEALTH_MARKER <src.tracking.logging.modules.training_health_emit.TRAINING_HEALTH_MARKER>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.training_health_emit.TRAINING_HEALTH_MARKER
    :summary:
    ```
````

### API

````{py:data} TRAINING_HEALTH_MARKER
:canonical: src.tracking.logging.modules.training_health_emit.TRAINING_HEALTH_MARKER
:value: >
   'TRAINING_HEALTH_START:'

```{autodoc2-docstring} src.tracking.logging.modules.training_health_emit.TRAINING_HEALTH_MARKER
```

````

````{py:function} emit_training_health_alert(code: str, severity: str, epoch: int, step: int, details: typing.Optional[typing.Dict[str, typing.Any]] = None, log_path: typing.Optional[str] = None, message: typing.Optional[str] = None) -> None
:canonical: src.tracking.logging.modules.training_health_emit.emit_training_health_alert

```{autodoc2-docstring} src.tracking.logging.modules.training_health_emit.emit_training_health_alert
```
````

````{py:function} _default_message(code: str) -> str
:canonical: src.tracking.logging.modules.training_health_emit._default_message

```{autodoc2-docstring} src.tracking.logging.modules.training_health_emit._default_message
```
````
