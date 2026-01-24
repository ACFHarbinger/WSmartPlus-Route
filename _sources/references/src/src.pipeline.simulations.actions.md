# {py:mod}`src.pipeline.simulations.actions`

```{py:module} src.pipeline.simulations.actions
```

```{autodoc2-docstring} src.pipeline.simulations.actions
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationAction <src.pipeline.simulations.actions.SimulationAction>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.SimulationAction
    :summary:
    ```
* - {py:obj}`FillAction <src.pipeline.simulations.actions.FillAction>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.FillAction
    :summary:
    ```
* - {py:obj}`PolicyExecutionAction <src.pipeline.simulations.actions.PolicyExecutionAction>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.PolicyExecutionAction
    :summary:
    ```
* - {py:obj}`CollectAction <src.pipeline.simulations.actions.CollectAction>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.CollectAction
    :summary:
    ```
* - {py:obj}`LogAction <src.pipeline.simulations.actions.LogAction>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.LogAction
    :summary:
    ```
````

### API

`````{py:class} SimulationAction
:canonical: src.pipeline.simulations.actions.SimulationAction

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.simulations.actions.SimulationAction
```

````{py:method} execute(context: typing.Any) -> None
:canonical: src.pipeline.simulations.actions.SimulationAction.execute
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.actions.SimulationAction.execute
```

````

`````

`````{py:class} FillAction
:canonical: src.pipeline.simulations.actions.FillAction

Bases: {py:obj}`src.pipeline.simulations.actions.SimulationAction`

```{autodoc2-docstring} src.pipeline.simulations.actions.FillAction
```

````{py:method} execute(context: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.simulations.actions.FillAction.execute

```{autodoc2-docstring} src.pipeline.simulations.actions.FillAction.execute
```

````

`````

`````{py:class} PolicyExecutionAction
:canonical: src.pipeline.simulations.actions.PolicyExecutionAction

Bases: {py:obj}`src.pipeline.simulations.actions.SimulationAction`

```{autodoc2-docstring} src.pipeline.simulations.actions.PolicyExecutionAction
```

````{py:method} execute(context: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.simulations.actions.PolicyExecutionAction.execute

```{autodoc2-docstring} src.pipeline.simulations.actions.PolicyExecutionAction.execute
```

````

`````

`````{py:class} CollectAction
:canonical: src.pipeline.simulations.actions.CollectAction

Bases: {py:obj}`src.pipeline.simulations.actions.SimulationAction`

```{autodoc2-docstring} src.pipeline.simulations.actions.CollectAction
```

````{py:method} execute(context: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.simulations.actions.CollectAction.execute

```{autodoc2-docstring} src.pipeline.simulations.actions.CollectAction.execute
```

````

`````

`````{py:class} LogAction
:canonical: src.pipeline.simulations.actions.LogAction

Bases: {py:obj}`src.pipeline.simulations.actions.SimulationAction`

```{autodoc2-docstring} src.pipeline.simulations.actions.LogAction
```

````{py:method} execute(context: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.simulations.actions.LogAction.execute

```{autodoc2-docstring} src.pipeline.simulations.actions.LogAction.execute
```

````

`````
