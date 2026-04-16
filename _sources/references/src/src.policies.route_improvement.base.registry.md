# {py:mod}`src.policies.route_improvement.base.registry`

```{py:module} src.policies.route_improvement.base.registry
```

```{autodoc2-docstring} src.policies.route_improvement.base.registry
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RouteImproverRegistry <src.policies.route_improvement.base.registry.RouteImproverRegistry>`
  - ```{autodoc2-docstring} src.policies.route_improvement.base.registry.RouteImproverRegistry
    :summary:
    ```
````

### API

`````{py:class} RouteImproverRegistry
:canonical: src.policies.route_improvement.base.registry.RouteImproverRegistry

```{autodoc2-docstring} src.policies.route_improvement.base.registry.RouteImproverRegistry
```

````{py:attribute} _strategies
:canonical: src.policies.route_improvement.base.registry.RouteImproverRegistry._strategies
:type: typing.Dict[str, typing.Type[logic.src.interfaces.route_improvement.IRouteImprovement]]
:value: >
   None

```{autodoc2-docstring} src.policies.route_improvement.base.registry.RouteImproverRegistry._strategies
```

````

````{py:method} register(name: str) -> typing.Callable
:canonical: src.policies.route_improvement.base.registry.RouteImproverRegistry.register
:classmethod:

```{autodoc2-docstring} src.policies.route_improvement.base.registry.RouteImproverRegistry.register
```

````

````{py:method} get_route_improver_class(name: str) -> typing.Optional[typing.Type[logic.src.interfaces.route_improvement.IRouteImprovement]]
:canonical: src.policies.route_improvement.base.registry.RouteImproverRegistry.get_route_improver_class
:classmethod:

```{autodoc2-docstring} src.policies.route_improvement.base.registry.RouteImproverRegistry.get_route_improver_class
```

````

`````
