# {py:mod}`src.utils.policy.wrappers`

```{py:module} src.utils.policy.wrappers
```

```{autodoc2-docstring} src.utils.policy.wrappers
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`initial_plan_greedy <src.utils.policy.wrappers.initial_plan_greedy>`
  - ```{autodoc2-docstring} src.utils.policy.wrappers.initial_plan_greedy
    :summary:
    ```
* - {py:obj}`_build_single_day_routes <src.utils.policy.wrappers._build_single_day_routes>`
  - ```{autodoc2-docstring} src.utils.policy.wrappers._build_single_day_routes
    :summary:
    ```
* - {py:obj}`greedy_day_route <src.utils.policy.wrappers.greedy_day_route>`
  - ```{autodoc2-docstring} src.utils.policy.wrappers.greedy_day_route
    :summary:
    ```
* - {py:obj}`two_opt <src.utils.policy.wrappers.two_opt>`
  - ```{autodoc2-docstring} src.utils.policy.wrappers.two_opt
    :summary:
    ```
````

### API

````{py:function} initial_plan_greedy(problem: logic.src.interfaces.context.problem_context.ProblemContext, rng: typing.Optional[numpy.random.Generator] = None, method: str = 'greedy') -> typing.List[typing.List[int]]
:canonical: src.utils.policy.wrappers.initial_plan_greedy

```{autodoc2-docstring} src.utils.policy.wrappers.initial_plan_greedy
```
````

````{py:function} _build_single_day_routes(problem: logic.src.interfaces.context.problem_context.ProblemContext, rng_stdlib: random.Random, rng_np: numpy.random.Generator, method: str) -> typing.List[typing.List[int]]
:canonical: src.utils.policy.wrappers._build_single_day_routes

```{autodoc2-docstring} src.utils.policy.wrappers._build_single_day_routes
```
````

````{py:function} greedy_day_route(problem: logic.src.interfaces.context.problem_context.ProblemContext, rng: numpy.random.Generator) -> typing.List[int]
:canonical: src.utils.policy.wrappers.greedy_day_route

```{autodoc2-docstring} src.utils.policy.wrappers.greedy_day_route
```
````

````{py:function} two_opt(route: typing.List[int], dist_matrix: numpy.ndarray, max_iter: int = 100) -> typing.List[int]
:canonical: src.utils.policy.wrappers.two_opt

```{autodoc2-docstring} src.utils.policy.wrappers.two_opt
```
````
