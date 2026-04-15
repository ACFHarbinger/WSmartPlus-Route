# {py:mod}`src.policies.hybrid_genetic_search.individual`

```{py:module} src.policies.hybrid_genetic_search.individual
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search.individual
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Individual <src.policies.hybrid_genetic_search.individual.Individual>`
  - ```{autodoc2-docstring} src.policies.hybrid_genetic_search.individual.Individual
    :summary:
    ```
````

### API

`````{py:class} Individual(giant_tour: typing.List[int], expand_pool: bool = False)
:canonical: src.policies.hybrid_genetic_search.individual.Individual

```{autodoc2-docstring} src.policies.hybrid_genetic_search.individual.Individual
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search.individual.Individual.__init__
```

````{py:property} routes
:canonical: src.policies.hybrid_genetic_search.individual.Individual.routes
:type: typing.List[typing.List[int]]

```{autodoc2-docstring} src.policies.hybrid_genetic_search.individual.Individual.routes
```

````

````{py:property} penalized_profit
:canonical: src.policies.hybrid_genetic_search.individual.Individual.penalized_profit
:type: float

```{autodoc2-docstring} src.policies.hybrid_genetic_search.individual.Individual.penalized_profit
```

````

````{py:method} get_visited_nodes() -> typing.Set[int]
:canonical: src.policies.hybrid_genetic_search.individual.Individual.get_visited_nodes

```{autodoc2-docstring} src.policies.hybrid_genetic_search.individual.Individual.get_visited_nodes
```

````

````{py:method} get_unvisited_nodes() -> typing.List[int]
:canonical: src.policies.hybrid_genetic_search.individual.Individual.get_unvisited_nodes

```{autodoc2-docstring} src.policies.hybrid_genetic_search.individual.Individual.get_unvisited_nodes
```

````

````{py:method} assert_invariants(expected_nodes: typing.Optional[typing.Set[int]] = None) -> None
:canonical: src.policies.hybrid_genetic_search.individual.Individual.assert_invariants

```{autodoc2-docstring} src.policies.hybrid_genetic_search.individual.Individual.assert_invariants
```

````

````{py:method} __lt__(other: src.policies.hybrid_genetic_search.individual.Individual) -> bool
:canonical: src.policies.hybrid_genetic_search.individual.Individual.__lt__

```{autodoc2-docstring} src.policies.hybrid_genetic_search.individual.Individual.__lt__
```

````

`````
