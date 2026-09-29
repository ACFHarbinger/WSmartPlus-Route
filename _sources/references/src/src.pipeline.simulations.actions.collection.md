# {py:mod}`src.pipeline.simulations.actions.collection`

```{py:module} src.pipeline.simulations.actions.collection
```

```{autodoc2-docstring} src.pipeline.simulations.actions.collection
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CollectAction <src.pipeline.simulations.actions.collection.CollectAction>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.collection.CollectAction
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`trip_loads <src.pipeline.simulations.actions.collection.trip_loads>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.collection.trip_loads
    :summary:
    ```
* - {py:obj}`capacity_report <src.pipeline.simulations.actions.collection.capacity_report>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.collection.capacity_report
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.simulations.actions.collection.logger>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.collection.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.simulations.actions.collection.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.pipeline.simulations.actions.collection.logger
```

````

````{py:function} trip_loads(tour: typing.List[int], fill_pct: numpy.ndarray) -> typing.List[float]
:canonical: src.pipeline.simulations.actions.collection.trip_loads

```{autodoc2-docstring} src.pipeline.simulations.actions.collection.trip_loads
```
````

````{py:function} capacity_report(tour: typing.List[int], bins: typing.Any, capacity_pct: float) -> typing.Tuple[typing.List[float], typing.List[float], int]
:canonical: src.pipeline.simulations.actions.collection.capacity_report

```{autodoc2-docstring} src.pipeline.simulations.actions.collection.capacity_report
```
````

`````{py:class} CollectAction
:canonical: src.pipeline.simulations.actions.collection.CollectAction

Bases: {py:obj}`src.pipeline.simulations.actions.base.SimulationAction`

```{autodoc2-docstring} src.pipeline.simulations.actions.collection.CollectAction
```

````{py:method} execute(context: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.simulations.actions.collection.CollectAction.execute

```{autodoc2-docstring} src.pipeline.simulations.actions.collection.CollectAction.execute
```

````

`````
