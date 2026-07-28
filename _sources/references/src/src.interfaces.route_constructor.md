# {py:mod}`src.interfaces.route_constructor`

```{py:module} src.interfaces.route_constructor
```

```{autodoc2-docstring} src.interfaces.route_constructor
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IRouteConstructor <src.interfaces.route_constructor.IRouteConstructor>`
  - ```{autodoc2-docstring} src.interfaces.route_constructor.IRouteConstructor
    :summary:
    ```
````

### API

`````{py:class} IRouteConstructor
:canonical: src.interfaces.route_constructor.IRouteConstructor

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.interfaces.route_constructor.IRouteConstructor
```

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.Union[typing.List[int], typing.List[typing.List[int]]], float, float, typing.Optional[logic.src.interfaces.context.search_context.SearchContext], typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]]
:canonical: src.interfaces.route_constructor.IRouteConstructor.execute
:abstractmethod:

```{autodoc2-docstring} src.interfaces.route_constructor.IRouteConstructor.execute
```

````

`````
