# {py:mod}`src.utils.policy.llh_pool`

```{py:module} src.utils.policy.llh_pool
```

```{autodoc2-docstring} src.utils.policy.llh_pool
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LLHPool <src.utils.policy.llh_pool.LLHPool>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool.LLHPool
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`h1_greedy_move <src.utils.policy.llh_pool.h1_greedy_move>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool.h1_greedy_move
    :summary:
    ```
* - {py:obj}`h2_random_drop <src.utils.policy.llh_pool.h2_random_drop>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool.h2_random_drop
    :summary:
    ```
* - {py:obj}`h3_two_opt <src.utils.policy.llh_pool.h3_two_opt>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool.h3_two_opt
    :summary:
    ```
* - {py:obj}`h4_swap_nodes <src.utils.policy.llh_pool.h4_swap_nodes>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool.h4_swap_nodes
    :summary:
    ```
* - {py:obj}`h5_relocate_node <src.utils.policy.llh_pool.h5_relocate_node>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool.h5_relocate_node
    :summary:
    ```
* - {py:obj}`h6_replace_node <src.utils.policy.llh_pool.h6_replace_node>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool.h6_replace_node
    :summary:
    ```
* - {py:obj}`h7_greedy_multi <src.utils.policy.llh_pool.h7_greedy_multi>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool.h7_greedy_multi
    :summary:
    ```
* - {py:obj}`h8_perturb_2opt <src.utils.policy.llh_pool.h8_perturb_2opt>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool.h8_perturb_2opt
    :summary:
    ```
* - {py:obj}`h9_worst_removal <src.utils.policy.llh_pool.h9_worst_removal>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool.h9_worst_removal
    :summary:
    ```
* - {py:obj}`h10_greedy_shuffle <src.utils.policy.llh_pool.h10_greedy_shuffle>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool.h10_greedy_shuffle
    :summary:
    ```
* - {py:obj}`_make_or_opt <src.utils.policy.llh_pool._make_or_opt>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool._make_or_opt
    :summary:
    ```
* - {py:obj}`_make_two_opt <src.utils.policy.llh_pool._make_two_opt>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool._make_two_opt
    :summary:
    ```
* - {py:obj}`_make_three_opt <src.utils.policy.llh_pool._make_three_opt>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool._make_three_opt
    :summary:
    ```
* - {py:obj}`_make_cross_day_move <src.utils.policy.llh_pool._make_cross_day_move>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool._make_cross_day_move
    :summary:
    ```
* - {py:obj}`_make_cross_day_swap <src.utils.policy.llh_pool._make_cross_day_swap>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool._make_cross_day_swap
    :summary:
    ```
* - {py:obj}`_make_day_merge_split <src.utils.policy.llh_pool._make_day_merge_split>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool._make_day_merge_split
    :summary:
    ```
* - {py:obj}`_make_double_bridge <src.utils.policy.llh_pool._make_double_bridge>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool._make_double_bridge
    :summary:
    ```
* - {py:obj}`_make_day_shuffle <src.utils.policy.llh_pool._make_day_shuffle>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool._make_day_shuffle
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Plan <src.utils.policy.llh_pool.Plan>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool.Plan
    :summary:
    ```
* - {py:obj}`LLH <src.utils.policy.llh_pool.LLH>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool.LLH
    :summary:
    ```
* - {py:obj}`INTRA_ROUTE_POOL <src.utils.policy.llh_pool.INTRA_ROUTE_POOL>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool.INTRA_ROUTE_POOL
    :summary:
    ```
* - {py:obj}`MULTI_PERIOD_LLH_POOL <src.utils.policy.llh_pool.MULTI_PERIOD_LLH_POOL>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool.MULTI_PERIOD_LLH_POOL
    :summary:
    ```
* - {py:obj}`LLH_POOL <src.utils.policy.llh_pool.LLH_POOL>`
  - ```{autodoc2-docstring} src.utils.policy.llh_pool.LLH_POOL
    :summary:
    ```
````

### API

````{py:data} Plan
:canonical: src.utils.policy.llh_pool.Plan
:value: >
   None

```{autodoc2-docstring} src.utils.policy.llh_pool.Plan
```

````

````{py:data} LLH
:canonical: src.utils.policy.llh_pool.LLH
:value: >
   None

```{autodoc2-docstring} src.utils.policy.llh_pool.LLH
```

````

````{py:function} h1_greedy_move(problem: logic.src.interfaces.context.problem_context.ProblemContext, route: typing.List[int], rng: numpy.random.Generator) -> typing.List[int]
:canonical: src.utils.policy.llh_pool.h1_greedy_move

```{autodoc2-docstring} src.utils.policy.llh_pool.h1_greedy_move
```
````

````{py:function} h2_random_drop(problem: logic.src.interfaces.context.problem_context.ProblemContext, route: typing.List[int], rng: numpy.random.Generator) -> typing.List[int]
:canonical: src.utils.policy.llh_pool.h2_random_drop

```{autodoc2-docstring} src.utils.policy.llh_pool.h2_random_drop
```
````

````{py:function} h3_two_opt(problem: logic.src.interfaces.context.problem_context.ProblemContext, route: typing.List[int], rng: numpy.random.Generator) -> typing.List[int]
:canonical: src.utils.policy.llh_pool.h3_two_opt

```{autodoc2-docstring} src.utils.policy.llh_pool.h3_two_opt
```
````

````{py:function} h4_swap_nodes(problem: logic.src.interfaces.context.problem_context.ProblemContext, route: typing.List[int], rng: numpy.random.Generator) -> typing.List[int]
:canonical: src.utils.policy.llh_pool.h4_swap_nodes

```{autodoc2-docstring} src.utils.policy.llh_pool.h4_swap_nodes
```
````

````{py:function} h5_relocate_node(problem: logic.src.interfaces.context.problem_context.ProblemContext, route: typing.List[int], rng: numpy.random.Generator) -> typing.List[int]
:canonical: src.utils.policy.llh_pool.h5_relocate_node

```{autodoc2-docstring} src.utils.policy.llh_pool.h5_relocate_node
```
````

````{py:function} h6_replace_node(problem: logic.src.interfaces.context.problem_context.ProblemContext, route: typing.List[int], rng: numpy.random.Generator) -> typing.List[int]
:canonical: src.utils.policy.llh_pool.h6_replace_node

```{autodoc2-docstring} src.utils.policy.llh_pool.h6_replace_node
```
````

````{py:function} h7_greedy_multi(problem: logic.src.interfaces.context.problem_context.ProblemContext, route: typing.List[int], rng: numpy.random.Generator) -> typing.List[int]
:canonical: src.utils.policy.llh_pool.h7_greedy_multi

```{autodoc2-docstring} src.utils.policy.llh_pool.h7_greedy_multi
```
````

````{py:function} h8_perturb_2opt(problem: logic.src.interfaces.context.problem_context.ProblemContext, route: typing.List[int], rng: numpy.random.Generator) -> typing.List[int]
:canonical: src.utils.policy.llh_pool.h8_perturb_2opt

```{autodoc2-docstring} src.utils.policy.llh_pool.h8_perturb_2opt
```
````

````{py:function} h9_worst_removal(problem: logic.src.interfaces.context.problem_context.ProblemContext, route: typing.List[int], rng: numpy.random.Generator) -> typing.List[int]
:canonical: src.utils.policy.llh_pool.h9_worst_removal

```{autodoc2-docstring} src.utils.policy.llh_pool.h9_worst_removal
```
````

````{py:function} h10_greedy_shuffle(problem: logic.src.interfaces.context.problem_context.ProblemContext, route: typing.List[int], rng: numpy.random.Generator) -> typing.List[int]
:canonical: src.utils.policy.llh_pool.h10_greedy_shuffle

```{autodoc2-docstring} src.utils.policy.llh_pool.h10_greedy_shuffle
```
````

````{py:function} _make_or_opt(chain_len: int) -> src.utils.policy.llh_pool.LLH
:canonical: src.utils.policy.llh_pool._make_or_opt

```{autodoc2-docstring} src.utils.policy.llh_pool._make_or_opt
```
````

````{py:function} _make_two_opt() -> src.utils.policy.llh_pool.LLH
:canonical: src.utils.policy.llh_pool._make_two_opt

```{autodoc2-docstring} src.utils.policy.llh_pool._make_two_opt
```
````

````{py:function} _make_three_opt() -> src.utils.policy.llh_pool.LLH
:canonical: src.utils.policy.llh_pool._make_three_opt

```{autodoc2-docstring} src.utils.policy.llh_pool._make_three_opt
```
````

````{py:function} _make_cross_day_move() -> src.utils.policy.llh_pool.LLH
:canonical: src.utils.policy.llh_pool._make_cross_day_move

```{autodoc2-docstring} src.utils.policy.llh_pool._make_cross_day_move
```
````

````{py:function} _make_cross_day_swap() -> src.utils.policy.llh_pool.LLH
:canonical: src.utils.policy.llh_pool._make_cross_day_swap

```{autodoc2-docstring} src.utils.policy.llh_pool._make_cross_day_swap
```
````

````{py:function} _make_day_merge_split() -> src.utils.policy.llh_pool.LLH
:canonical: src.utils.policy.llh_pool._make_day_merge_split

```{autodoc2-docstring} src.utils.policy.llh_pool._make_day_merge_split
```
````

````{py:function} _make_double_bridge() -> src.utils.policy.llh_pool.LLH
:canonical: src.utils.policy.llh_pool._make_double_bridge

```{autodoc2-docstring} src.utils.policy.llh_pool._make_double_bridge
```
````

````{py:function} _make_day_shuffle() -> src.utils.policy.llh_pool.LLH
:canonical: src.utils.policy.llh_pool._make_day_shuffle

```{autodoc2-docstring} src.utils.policy.llh_pool._make_day_shuffle
```
````

````{py:data} INTRA_ROUTE_POOL
:canonical: src.utils.policy.llh_pool.INTRA_ROUTE_POOL
:value: >
   None

```{autodoc2-docstring} src.utils.policy.llh_pool.INTRA_ROUTE_POOL
```

````

````{py:data} MULTI_PERIOD_LLH_POOL
:canonical: src.utils.policy.llh_pool.MULTI_PERIOD_LLH_POOL
:type: typing.List[typing.Tuple[str, src.utils.policy.llh_pool.LLH]]
:value: >
   [('or_opt_1',), ('or_opt_2',), ('or_opt_3',), ('two_opt',), ('three_opt',), ('cross_day_move',), ('c...

```{autodoc2-docstring} src.utils.policy.llh_pool.MULTI_PERIOD_LLH_POOL
```

````

````{py:data} LLH_POOL
:canonical: src.utils.policy.llh_pool.LLH_POOL
:value: >
   None

```{autodoc2-docstring} src.utils.policy.llh_pool.LLH_POOL
```

````

`````{py:class} LLHPool
:canonical: src.utils.policy.llh_pool.LLHPool

```{autodoc2-docstring} src.utils.policy.llh_pool.LLHPool
```

````{py:method} llh_swap(plan: typing.List[typing.List[typing.List[int]]], problem: logic.src.interfaces.context.problem_context.ProblemContext, rng: random.Random) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.utils.policy.llh_pool.LLHPool.llh_swap
:staticmethod:

```{autodoc2-docstring} src.utils.policy.llh_pool.LLHPool.llh_swap
```

````

````{py:method} llh_drop_add(plan: typing.List[typing.List[typing.List[int]]], problem: logic.src.interfaces.context.problem_context.ProblemContext, rng: random.Random) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.utils.policy.llh_pool.LLHPool.llh_drop_add
:staticmethod:

```{autodoc2-docstring} src.utils.policy.llh_pool.LLHPool.llh_drop_add
```

````

````{py:method} llh_2opt_all(plan: typing.List[typing.List[typing.List[int]]], problem: logic.src.interfaces.context.problem_context.ProblemContext, rng: random.Random) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.utils.policy.llh_pool.LLHPool.llh_2opt_all
:staticmethod:

```{autodoc2-docstring} src.utils.policy.llh_pool.LLHPool.llh_2opt_all
```

````

````{py:method} llh_ruin_recreate_day(plan: typing.List[typing.List[typing.List[int]]], problem: logic.src.interfaces.context.problem_context.ProblemContext, rng: random.Random) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.utils.policy.llh_pool.LLHPool.llh_ruin_recreate_day
:staticmethod:

```{autodoc2-docstring} src.utils.policy.llh_pool.LLHPool.llh_ruin_recreate_day
```

````

````{py:method} get_all() -> typing.List[typing.Callable]
:canonical: src.utils.policy.llh_pool.LLHPool.get_all
:classmethod:

```{autodoc2-docstring} src.utils.policy.llh_pool.LLHPool.get_all
```

````

`````
