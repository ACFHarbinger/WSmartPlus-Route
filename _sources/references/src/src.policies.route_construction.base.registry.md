# {py:mod}`src.policies.route_construction.base.registry`

```{py:module} src.policies.route_construction.base.registry
```

```{autodoc2-docstring} src.policies.route_construction.base.registry
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RouteConstructorRegistry <src.policies.route_construction.base.registry.RouteConstructorRegistry>`
  - ```{autodoc2-docstring} src.policies.route_construction.base.registry.RouteConstructorRegistry
    :summary:
    ```
````

### API

`````{py:class} RouteConstructorRegistry
:canonical: src.policies.route_construction.base.registry.RouteConstructorRegistry

```{autodoc2-docstring} src.policies.route_construction.base.registry.RouteConstructorRegistry
```

````{py:attribute} _registry
:canonical: src.policies.route_construction.base.registry.RouteConstructorRegistry._registry
:type: typing.Dict[str, typing.Type[logic.src.interfaces.route_constructor.IRouteConstructor]]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.base.registry.RouteConstructorRegistry._registry
```

````

````{py:method} register(name: str) -> typing.Callable
:canonical: src.policies.route_construction.base.registry.RouteConstructorRegistry.register
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.base.registry.RouteConstructorRegistry.register
```

````

````{py:method} get(name: str) -> typing.Optional[typing.Type[logic.src.interfaces.route_constructor.IRouteConstructor]]
:canonical: src.policies.route_construction.base.registry.RouteConstructorRegistry.get
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.base.registry.RouteConstructorRegistry.get
```

````

````{py:method} list_route_constructors() -> typing.List[str]
:canonical: src.policies.route_construction.base.registry.RouteConstructorRegistry.list_route_constructors
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.base.registry.RouteConstructorRegistry.list_route_constructors
```

````

`````
