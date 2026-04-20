# {py:mod}`src.policies.helpers.operators.perturbation_shaking.cross_day`

```{py:module} src.policies.helpers.operators.perturbation_shaking.cross_day
```

```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.cross_day
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`recompute_cascade_from <src.policies.helpers.operators.perturbation_shaking.cross_day.recompute_cascade_from>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.cross_day.recompute_cascade_from
    :summary:
    ```
* - {py:obj}`cross_day_move <src.policies.helpers.operators.perturbation_shaking.cross_day.cross_day_move>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.cross_day.cross_day_move
    :summary:
    ```
* - {py:obj}`cross_day_swap <src.policies.helpers.operators.perturbation_shaking.cross_day.cross_day_swap>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.cross_day.cross_day_swap
    :summary:
    ```
* - {py:obj}`day_merge_split <src.policies.helpers.operators.perturbation_shaking.cross_day.day_merge_split>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.cross_day.day_merge_split
    :summary:
    ```
````

### API

````{py:function} recompute_cascade_from(plan: typing.List[typing.List[int]], problem: logic.src.interfaces.context.problem_context.ProblemContext, start_day: int) -> typing.Tuple[typing.List[typing.List[int]], logic.src.interfaces.context.problem_context.ProblemContext]
:canonical: src.policies.helpers.operators.perturbation_shaking.cross_day.recompute_cascade_from

```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.cross_day.recompute_cascade_from
```
````

````{py:function} cross_day_move(plan: typing.List[typing.List[int]], problem: logic.src.interfaces.context.problem_context.ProblemContext, bin_id: int, from_day: int, to_day: int) -> typing.Tuple[typing.List[typing.List[int]], int]
:canonical: src.policies.helpers.operators.perturbation_shaking.cross_day.cross_day_move

```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.cross_day.cross_day_move
```
````

````{py:function} cross_day_swap(plan: typing.List[typing.List[int]], problem: logic.src.interfaces.context.problem_context.ProblemContext, bin_i: int, day_i: int, bin_j: int, day_j: int) -> typing.Tuple[typing.List[typing.List[int]], int]
:canonical: src.policies.helpers.operators.perturbation_shaking.cross_day.cross_day_swap

```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.cross_day.cross_day_swap
```
````

````{py:function} day_merge_split(plan: typing.List[typing.List[int]], problem: logic.src.interfaces.context.problem_context.ProblemContext, rng: numpy.random.Generator) -> typing.Tuple[typing.List[typing.List[int]], int]
:canonical: src.policies.helpers.operators.perturbation_shaking.cross_day.day_merge_split

```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.cross_day.day_merge_split
```
````
