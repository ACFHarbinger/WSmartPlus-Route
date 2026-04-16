# {py:mod}`src.policies.route_construction.base.factory`

```{py:module} src.policies.route_construction.base.factory
```

```{autodoc2-docstring} src.policies.route_construction.base.factory
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RouteConstructorFactory <src.policies.route_construction.base.factory.RouteConstructorFactory>`
  - ```{autodoc2-docstring} src.policies.route_construction.base.factory.RouteConstructorFactory
    :summary:
    ```
````

### API

`````{py:class} RouteConstructorFactory
:canonical: src.policies.route_construction.base.factory.RouteConstructorFactory

```{autodoc2-docstring} src.policies.route_construction.base.factory.RouteConstructorFactory
```

````{py:attribute} _registered
:canonical: src.policies.route_construction.base.factory.RouteConstructorFactory._registered
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.base.factory.RouteConstructorFactory._registered
```

````

````{py:method} ensure_registered() -> None
:canonical: src.policies.route_construction.base.factory.RouteConstructorFactory.ensure_registered
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.base.factory.RouteConstructorFactory.ensure_registered
```

````

````{py:method} get_adapter(name: str, config: typing.Optional[dict] = None, engine: typing.Optional[str] = None, threshold: typing.Optional[float] = None, **kwargs: typing.Any) -> logic.src.interfaces.route_constructor.IRouteConstructor
:canonical: src.policies.route_construction.base.factory.RouteConstructorFactory.get_adapter
:staticmethod:

```{autodoc2-docstring} src.policies.route_construction.base.factory.RouteConstructorFactory.get_adapter
```

````

`````
