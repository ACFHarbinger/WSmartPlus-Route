# {py:mod}`src.pipeline.simulations.states.finishing`

```{py:module} src.pipeline.simulations.states.finishing
```

```{autodoc2-docstring} src.pipeline.simulations.states.finishing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FinishingState <src.pipeline.simulations.states.finishing.FinishingState>`
  - ```{autodoc2-docstring} src.pipeline.simulations.states.finishing.FinishingState
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_log_result_artifacts <src.pipeline.simulations.states.finishing._log_result_artifacts>`
  - ```{autodoc2-docstring} src.pipeline.simulations.states.finishing._log_result_artifacts
    :summary:
    ```
````

### API

`````{py:class} FinishingState
:canonical: src.pipeline.simulations.states.finishing.FinishingState

Bases: {py:obj}`src.pipeline.simulations.states.base.SimState`

```{autodoc2-docstring} src.pipeline.simulations.states.finishing.FinishingState
```

````{py:method} handle(ctx: src.pipeline.simulations.states.base.SimulationContext) -> None
:canonical: src.pipeline.simulations.states.finishing.FinishingState.handle

```{autodoc2-docstring} src.pipeline.simulations.states.finishing.FinishingState.handle
```

````

`````

````{py:function} _log_result_artifacts(ctx: typing.Any, sim: typing.Any, log_path: str, daily_log_path: str) -> None
:canonical: src.pipeline.simulations.states.finishing._log_result_artifacts

```{autodoc2-docstring} src.pipeline.simulations.states.finishing._log_result_artifacts
```
````
