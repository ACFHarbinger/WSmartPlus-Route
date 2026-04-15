# {py:mod}`src.policies.other.operators.crossover.edge_recombination`

```{py:module} src.policies.other.operators.crossover.edge_recombination
```

```{autodoc2-docstring} src.policies.other.operators.crossover.edge_recombination
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`edge_recombination_crossover <src.policies.other.operators.crossover.edge_recombination.edge_recombination_crossover>`
  - ```{autodoc2-docstring} src.policies.other.operators.crossover.edge_recombination.edge_recombination_crossover
    :summary:
    ```
* - {py:obj}`_get_physical_edges <src.policies.other.operators.crossover.edge_recombination._get_physical_edges>`
  - ```{autodoc2-docstring} src.policies.other.operators.crossover.edge_recombination._get_physical_edges
    :summary:
    ```
* - {py:obj}`capacity_aware_erx <src.policies.other.operators.crossover.edge_recombination.capacity_aware_erx>`
  - ```{autodoc2-docstring} src.policies.other.operators.crossover.edge_recombination.capacity_aware_erx
    :summary:
    ```
````

### API

````{py:function} edge_recombination_crossover(p1: logic.src.policies.hybrid_genetic_search.individual.Individual, p2: logic.src.policies.hybrid_genetic_search.individual.Individual, rng: typing.Optional[random.Random] = None, mandatory_nodes: typing.Optional[typing.List[int]] = None) -> logic.src.policies.hybrid_genetic_search.individual.Individual
:canonical: src.policies.other.operators.crossover.edge_recombination.edge_recombination_crossover

```{autodoc2-docstring} src.policies.other.operators.crossover.edge_recombination.edge_recombination_crossover
```
````

````{py:function} _get_physical_edges(routes: typing.List[typing.List[int]]) -> typing.Set[typing.Tuple[int, int]]
:canonical: src.policies.other.operators.crossover.edge_recombination._get_physical_edges

```{autodoc2-docstring} src.policies.other.operators.crossover.edge_recombination._get_physical_edges
```
````

````{py:function} capacity_aware_erx(p1: logic.src.policies.hybrid_genetic_search.individual.Individual, p2: logic.src.policies.hybrid_genetic_search.individual.Individual, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float = 1.0, C: float = 1.0, rng: typing.Optional[random.Random] = None, mandatory_nodes: typing.Optional[typing.List[int]] = None) -> logic.src.policies.hybrid_genetic_search.individual.Individual
:canonical: src.policies.other.operators.crossover.edge_recombination.capacity_aware_erx

```{autodoc2-docstring} src.policies.other.operators.crossover.edge_recombination.capacity_aware_erx
```
````
