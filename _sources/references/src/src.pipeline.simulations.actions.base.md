# {py:mod}`src.pipeline.simulations.actions.base`

```{py:module} src.pipeline.simulations.actions.base
```

```{autodoc2-docstring} src.pipeline.simulations.actions.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationAction <src.pipeline.simulations.actions.base.SimulationAction>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base.SimulationAction
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_flatten_config <src.pipeline.simulations.actions.base._flatten_config>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._flatten_config
    :summary:
    ```
````

### API

````{py:function} _flatten_config(cfg: typing.Any) -> dict
:canonical: src.pipeline.simulations.actions.base._flatten_config

```{autodoc2-docstring} src.pipeline.simulations.actions.base._flatten_config
```
````

`````{py:class} SimulationAction
:canonical: src.pipeline.simulations.actions.base.SimulationAction

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.simulations.actions.base.SimulationAction
```

````{py:method} execute(context: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.simulations.actions.base.SimulationAction.execute
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.actions.base.SimulationAction.execute
```

````

`````
