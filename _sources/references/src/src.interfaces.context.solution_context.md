# {py:mod}`src.interfaces.context.solution_context`

```{py:module} src.interfaces.context.solution_context
```

```{autodoc2-docstring} src.interfaces.context.solution_context
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SolutionContext <src.interfaces.context.solution_context.SolutionContext>`
  - ```{autodoc2-docstring} src.interfaces.context.solution_context.SolutionContext
    :summary:
    ```
````

### API

`````{py:class} SolutionContext
:canonical: src.interfaces.context.solution_context.SolutionContext

```{autodoc2-docstring} src.interfaces.context.solution_context.SolutionContext
```

````{py:attribute} routes
:canonical: src.interfaces.context.solution_context.SolutionContext.routes
:type: typing.List[typing.List[int]]
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.solution_context.SolutionContext.routes
```

````

````{py:attribute} profits
:canonical: src.interfaces.context.solution_context.SolutionContext.profits
:type: typing.List[float]
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.solution_context.SolutionContext.profits
```

````

````{py:attribute} costs
:canonical: src.interfaces.context.solution_context.SolutionContext.costs
:type: typing.List[float]
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.solution_context.SolutionContext.costs
```

````

````{py:attribute} total_profit
:canonical: src.interfaces.context.solution_context.SolutionContext.total_profit
:type: float
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.solution_context.SolutionContext.total_profit
```

````

````{py:attribute} total_cost
:canonical: src.interfaces.context.solution_context.SolutionContext.total_cost
:type: float
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.solution_context.SolutionContext.total_cost
```

````

````{py:attribute} metadata
:canonical: src.interfaces.context.solution_context.SolutionContext.metadata
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} src.interfaces.context.solution_context.SolutionContext.metadata
```

````

````{py:method} empty() -> src.interfaces.context.solution_context.SolutionContext
:canonical: src.interfaces.context.solution_context.SolutionContext.empty
:classmethod:

```{autodoc2-docstring} src.interfaces.context.solution_context.SolutionContext.empty
```

````

````{py:method} from_single_route(route: typing.List[int], profit: float, cost: float, metadata: typing.Optional[dict] = None) -> src.interfaces.context.solution_context.SolutionContext
:canonical: src.interfaces.context.solution_context.SolutionContext.from_single_route
:classmethod:

```{autodoc2-docstring} src.interfaces.context.solution_context.SolutionContext.from_single_route
```

````

````{py:method} from_problem(problem: src.interfaces.context.problem_context.ProblemContext, route: typing.List[int], metadata: typing.Optional[dict] = None) -> src.interfaces.context.solution_context.SolutionContext
:canonical: src.interfaces.context.solution_context.SolutionContext.from_problem
:classmethod:

```{autodoc2-docstring} src.interfaces.context.solution_context.SolutionContext.from_problem
```

````

````{py:method} from_multi_route(problem: src.interfaces.context.problem_context.ProblemContext, routes: typing.List[typing.List[int]], metadata: typing.Optional[dict] = None) -> src.interfaces.context.solution_context.SolutionContext
:canonical: src.interfaces.context.solution_context.SolutionContext.from_multi_route
:classmethod:

```{autodoc2-docstring} src.interfaces.context.solution_context.SolutionContext.from_multi_route
```

````

````{py:method} to_flat_tour() -> typing.List[int]
:canonical: src.interfaces.context.solution_context.SolutionContext.to_flat_tour

```{autodoc2-docstring} src.interfaces.context.solution_context.SolutionContext.to_flat_tour
```

````

`````
