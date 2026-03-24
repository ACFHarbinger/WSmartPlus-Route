# {py:mod}`src.policies.lin_kernighan_helsgaun_three.kopt_topologies`

```{py:module} src.policies.lin_kernighan_helsgaun_three.kopt_topologies
```

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_generate_all_perfect_matchings <src.policies.lin_kernighan_helsgaun_three.kopt_topologies._generate_all_perfect_matchings>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies._generate_all_perfect_matchings
    :summary:
    ```
* - {py:obj}`_is_hamiltonian_cycle <src.policies.lin_kernighan_helsgaun_three.kopt_topologies._is_hamiltonian_cycle>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies._is_hamiltonian_cycle
    :summary:
    ```
* - {py:obj}`_is_trivial_move <src.policies.lin_kernighan_helsgaun_three.kopt_topologies._is_trivial_move>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies._is_trivial_move
    :summary:
    ```
* - {py:obj}`_is_non_sequential_kopt <src.policies.lin_kernighan_helsgaun_three.kopt_topologies._is_non_sequential_kopt>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies._is_non_sequential_kopt
    :summary:
    ```
* - {py:obj}`_generate_valid_kopt_topologies <src.policies.lin_kernighan_helsgaun_three.kopt_topologies._generate_valid_kopt_topologies>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies._generate_valid_kopt_topologies
    :summary:
    ```
* - {py:obj}`verify_topology_counts <src.policies.lin_kernighan_helsgaun_three.kopt_topologies.verify_topology_counts>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies.verify_topology_counts
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EXHAUSTIVE_2OPT_CASES <src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXHAUSTIVE_2OPT_CASES>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXHAUSTIVE_2OPT_CASES
    :summary:
    ```
* - {py:obj}`EXHAUSTIVE_3OPT_CASES <src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXHAUSTIVE_3OPT_CASES>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXHAUSTIVE_3OPT_CASES
    :summary:
    ```
* - {py:obj}`EXHAUSTIVE_4OPT_CASES <src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXHAUSTIVE_4OPT_CASES>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXHAUSTIVE_4OPT_CASES
    :summary:
    ```
* - {py:obj}`EXHAUSTIVE_5OPT_CASES <src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXHAUSTIVE_5OPT_CASES>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXHAUSTIVE_5OPT_CASES
    :summary:
    ```
* - {py:obj}`EXPECTED_COUNTS <src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXPECTED_COUNTS>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXPECTED_COUNTS
    :summary:
    ```
````

### API

````{py:function} _generate_all_perfect_matchings(nodes: typing.List[int]) -> typing.Iterator[typing.List[typing.Tuple[int, int]]]
:canonical: src.policies.lin_kernighan_helsgaun_three.kopt_topologies._generate_all_perfect_matchings

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies._generate_all_perfect_matchings
```
````

````{py:function} _is_hamiltonian_cycle(num_nodes: int, segments: typing.List[typing.Tuple[int, int]], added_edges: typing.List[typing.Tuple[int, int]]) -> bool
:canonical: src.policies.lin_kernighan_helsgaun_three.kopt_topologies._is_hamiltonian_cycle

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies._is_hamiltonian_cycle
```
````

````{py:function} _is_trivial_move(k: int, added_edges: typing.List[typing.Tuple[int, int]]) -> bool
:canonical: src.policies.lin_kernighan_helsgaun_three.kopt_topologies._is_trivial_move

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies._is_trivial_move
```
````

````{py:function} _is_non_sequential_kopt(k: int, added_edges: typing.List[typing.Tuple[int, int]]) -> bool
:canonical: src.policies.lin_kernighan_helsgaun_three.kopt_topologies._is_non_sequential_kopt

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies._is_non_sequential_kopt
```
````

````{py:function} _generate_valid_kopt_topologies(k: int) -> typing.List[typing.List[typing.Tuple[int, int]]]
:canonical: src.policies.lin_kernighan_helsgaun_three.kopt_topologies._generate_valid_kopt_topologies

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies._generate_valid_kopt_topologies
```
````

````{py:data} EXHAUSTIVE_2OPT_CASES
:canonical: src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXHAUSTIVE_2OPT_CASES
:type: typing.List[typing.List[typing.Tuple[int, int]]]
:value: >
   '_generate_valid_kopt_topologies(...)'

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXHAUSTIVE_2OPT_CASES
```

````

````{py:data} EXHAUSTIVE_3OPT_CASES
:canonical: src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXHAUSTIVE_3OPT_CASES
:type: typing.List[typing.List[typing.Tuple[int, int]]]
:value: >
   '_generate_valid_kopt_topologies(...)'

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXHAUSTIVE_3OPT_CASES
```

````

````{py:data} EXHAUSTIVE_4OPT_CASES
:canonical: src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXHAUSTIVE_4OPT_CASES
:type: typing.List[typing.List[typing.Tuple[int, int]]]
:value: >
   '_generate_valid_kopt_topologies(...)'

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXHAUSTIVE_4OPT_CASES
```

````

````{py:data} EXHAUSTIVE_5OPT_CASES
:canonical: src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXHAUSTIVE_5OPT_CASES
:type: typing.List[typing.List[typing.Tuple[int, int]]]
:value: >
   '_generate_valid_kopt_topologies(...)'

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXHAUSTIVE_5OPT_CASES
```

````

````{py:data} EXPECTED_COUNTS
:canonical: src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXPECTED_COUNTS
:type: typing.Dict[int, int]
:value: >
   None

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies.EXPECTED_COUNTS
```

````

````{py:function} verify_topology_counts() -> bool
:canonical: src.policies.lin_kernighan_helsgaun_three.kopt_topologies.verify_topology_counts

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.kopt_topologies.verify_topology_counts
```
````
