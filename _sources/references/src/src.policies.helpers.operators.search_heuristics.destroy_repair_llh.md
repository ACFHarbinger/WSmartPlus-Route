# {py:mod}`src.policies.helpers.operators.search_heuristics.destroy_repair_llh`

```{py:module} src.policies.helpers.operators.search_heuristics.destroy_repair_llh
```

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_insert_greedy <src.policies.helpers.operators.search_heuristics.destroy_repair_llh._insert_greedy>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh._insert_greedy
    :summary:
    ```
* - {py:obj}`_insert_regret_2 <src.policies.helpers.operators.search_heuristics.destroy_repair_llh._insert_regret_2>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh._insert_regret_2
    :summary:
    ```
* - {py:obj}`_remove_worst <src.policies.helpers.operators.search_heuristics.destroy_repair_llh._remove_worst>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh._remove_worst
    :summary:
    ```
* - {py:obj}`llh_random_greedy <src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_random_greedy>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_random_greedy
    :summary:
    ```
* - {py:obj}`llh_worst_regret_2 <src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_worst_regret_2>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_worst_regret_2
    :summary:
    ```
* - {py:obj}`llh_cluster_greedy <src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_cluster_greedy>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_cluster_greedy
    :summary:
    ```
* - {py:obj}`llh_worst_greedy <src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_worst_greedy>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_worst_greedy
    :summary:
    ```
* - {py:obj}`llh_random_regret_2 <src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_random_regret_2>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_random_regret_2
    :summary:
    ```
* - {py:obj}`routes_total_distance <src.policies.helpers.operators.search_heuristics.destroy_repair_llh.routes_total_distance>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh.routes_total_distance
    :summary:
    ```
* - {py:obj}`routes_net_profit <src.policies.helpers.operators.search_heuristics.destroy_repair_llh.routes_net_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh.routes_net_profit
    :summary:
    ```
* - {py:obj}`build_greedy_initial_routes <src.policies.helpers.operators.search_heuristics.destroy_repair_llh.build_greedy_initial_routes>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh.build_greedy_initial_routes
    :summary:
    ```
````

### API

````{py:function} _insert_greedy(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int], expand_pool: bool, profit_aware: bool) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.search_heuristics.destroy_repair_llh._insert_greedy

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh._insert_greedy
```
````

````{py:function} _insert_regret_2(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int], expand_pool: bool, profit_aware: bool) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.search_heuristics.destroy_repair_llh._insert_regret_2

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh._insert_regret_2
```
````

````{py:function} _remove_worst(routes: typing.List[typing.List[int]], n: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float, C: float, profit_aware: bool) -> tuple
:canonical: src.policies.helpers.operators.search_heuristics.destroy_repair_llh._remove_worst

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh._remove_worst
```
````

````{py:function} llh_random_greedy(routes: typing.List[typing.List[int]], n: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = True, profit_aware: bool = False, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_random_greedy

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_random_greedy
```
````

````{py:function} llh_worst_regret_2(routes: typing.List[typing.List[int]], n: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = True, profit_aware: bool = False, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_worst_regret_2

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_worst_regret_2
```
````

````{py:function} llh_cluster_greedy(routes: typing.List[typing.List[int]], n: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = True, profit_aware: bool = False, rng: typing.Optional[random.Random] = None, nodes: typing.Optional[typing.List[int]] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_cluster_greedy

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_cluster_greedy
```
````

````{py:function} llh_worst_greedy(routes: typing.List[typing.List[int]], n: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = True, profit_aware: bool = False, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_worst_greedy

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_worst_greedy
```
````

````{py:function} llh_random_regret_2(routes: typing.List[typing.List[int]], n: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = True, profit_aware: bool = False, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_random_regret_2

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh.llh_random_regret_2
```
````

````{py:function} routes_total_distance(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray) -> float
:canonical: src.policies.helpers.operators.search_heuristics.destroy_repair_llh.routes_total_distance

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh.routes_total_distance
```
````

````{py:function} routes_net_profit(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float, C: float) -> float
:canonical: src.policies.helpers.operators.search_heuristics.destroy_repair_llh.routes_net_profit

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh.routes_net_profit
```
````

````{py:function} build_greedy_initial_routes(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.search_heuristics.destroy_repair_llh.build_greedy_initial_routes

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.destroy_repair_llh.build_greedy_initial_routes
```
````
