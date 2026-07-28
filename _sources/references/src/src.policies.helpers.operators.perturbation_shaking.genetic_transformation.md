# {py:mod}`src.policies.helpers.operators.perturbation_shaking.genetic_transformation`

```{py:module} src.policies.helpers.operators.perturbation_shaking.genetic_transformation
```

```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.genetic_transformation
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`genetic_transformation <src.policies.helpers.operators.perturbation_shaking.genetic_transformation.genetic_transformation>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.genetic_transformation.genetic_transformation
    :summary:
    ```
* - {py:obj}`genetic_transformation_profit <src.policies.helpers.operators.perturbation_shaking.genetic_transformation.genetic_transformation_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.genetic_transformation.genetic_transformation_profit
    :summary:
    ```
* - {py:obj}`_extract_edges <src.policies.helpers.operators.perturbation_shaking.genetic_transformation._extract_edges>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.genetic_transformation._extract_edges
    :summary:
    ```
````

### API

````{py:function} genetic_transformation(routes: typing.List[typing.List[int]], elite_solution: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.perturbation_shaking.genetic_transformation.genetic_transformation

```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.genetic_transformation.genetic_transformation
```
````

````{py:function} genetic_transformation_profit(routes: typing.List[typing.List[int]], elite_solution: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.perturbation_shaking.genetic_transformation.genetic_transformation_profit

```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.genetic_transformation.genetic_transformation_profit
```
````

````{py:function} _extract_edges(solution: typing.List[typing.List[int]]) -> typing.Set[typing.Tuple[int, int]]
:canonical: src.policies.helpers.operators.perturbation_shaking.genetic_transformation._extract_edges

```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.genetic_transformation._extract_edges
```
````
