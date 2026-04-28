# {py:mod}`src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool`

```{py:module} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VRPPRoute <src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute
    :summary:
    ```
* - {py:obj}`RoutePool <src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool
    :summary:
    ```
````

### API

`````{py:class} VRPPRoute
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute
```

````{py:attribute} nodes
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute.nodes
:type: typing.List[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute.nodes
```

````

````{py:attribute} profit
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute.profit
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute.profit
```

````

````{py:attribute} revenue
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute.revenue
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute.revenue
```

````

````{py:attribute} cost
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute.cost
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute.cost
```

````

````{py:attribute} load
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute.load
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute.load
```

````

````{py:attribute} source
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute.source
:type: str
:value: >
   'unknown'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute.source
```

````

````{py:method} canonical_key() -> typing.FrozenSet[int]
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute.canonical_key

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute.canonical_key
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute.__repr__

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute.__repr__
```

````

`````

`````{py:class} RoutePool()
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.__init__
```

````{py:method} add(route: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute) -> bool
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.add

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.add
```

````

````{py:method} add_all(routes: typing.List[src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute]) -> int
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.add_all

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.add_all
```

````

````{py:method} filter_feasible(capacity: float) -> None
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.filter_feasible

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.filter_feasible
```

````

````{py:method} routes() -> typing.List[src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute]
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.routes

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.routes
```

````

````{py:method} best() -> typing.Optional[src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute]
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.best

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.best
```

````

````{py:method} __len__() -> int
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.__len__

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.__len__
```

````

````{py:method} __iter__() -> typing.Iterator[src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute]
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.__iter__

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.__iter__
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.__repr__

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool.__repr__
```

````

`````
