# {py:mod}`src.tracking.logging.modules.failure_emit`

```{py:module} src.tracking.logging.modules.failure_emit
```

```{autodoc2-docstring} src.tracking.logging.modules.failure_emit
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`emit_sim_failure_summary <src.tracking.logging.modules.failure_emit.emit_sim_failure_summary>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.failure_emit.emit_sim_failure_summary
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SIM_FAILURE_MARKER <src.tracking.logging.modules.failure_emit.SIM_FAILURE_MARKER>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.failure_emit.SIM_FAILURE_MARKER
    :summary:
    ```
````

### API

````{py:data} SIM_FAILURE_MARKER
:canonical: src.tracking.logging.modules.failure_emit.SIM_FAILURE_MARKER
:value: >
   'SIM_FAILURE_START:'

```{autodoc2-docstring} src.tracking.logging.modules.failure_emit.SIM_FAILURE_MARKER
```

````

````{py:function} emit_sim_failure_summary(summary: typing.Dict[str, typing.Any], policy: str, sample_idx: int, day: int, log_path: typing.Optional[str] = None, lock: typing.Optional[threading.Lock] = None) -> None
:canonical: src.tracking.logging.modules.failure_emit.emit_sim_failure_summary

```{autodoc2-docstring} src.tracking.logging.modules.failure_emit.emit_sim_failure_summary
```
````
