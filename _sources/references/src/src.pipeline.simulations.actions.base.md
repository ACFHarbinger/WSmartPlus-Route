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

* - {py:obj}`_find_key <src.pipeline.simulations.actions.base._find_key>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._find_key
    :summary:
    ```
* - {py:obj}`_flatten_config <src.pipeline.simulations.actions.base._flatten_config>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._flatten_config
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_EXPANDED_KEYS <src.pipeline.simulations.actions.base._EXPANDED_KEYS>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.base._EXPANDED_KEYS
    :summary:
    ```
````

### API

````{py:function} _find_key(d: typing.Any, target_key: str) -> typing.Any
:canonical: src.pipeline.simulations.actions.base._find_key

```{autodoc2-docstring} src.pipeline.simulations.actions.base._find_key
```
````

````{py:data} _EXPANDED_KEYS
:canonical: src.pipeline.simulations.actions.base._EXPANDED_KEYS
:value: >
   ('mandatory_selection', 'route_improvement', 'acceptance_criteria', 'acceptance_criterion')

```{autodoc2-docstring} src.pipeline.simulations.actions.base._EXPANDED_KEYS
```

````

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
