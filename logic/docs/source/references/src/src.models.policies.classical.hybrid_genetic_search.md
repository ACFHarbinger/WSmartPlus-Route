# {py:mod}`src.models.policies.classical.hybrid_genetic_search`

```{py:module} src.models.policies.classical.hybrid_genetic_search
```

```{autodoc2-docstring} src.models.policies.classical.hybrid_genetic_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedPopulation <src.models.policies.classical.hybrid_genetic_search.VectorizedPopulation>`
  - ```{autodoc2-docstring} src.models.policies.classical.hybrid_genetic_search.VectorizedPopulation
    :summary:
    ```
* - {py:obj}`VectorizedHGS <src.models.policies.classical.hybrid_genetic_search.VectorizedHGS>`
  - ```{autodoc2-docstring} src.models.policies.classical.hybrid_genetic_search.VectorizedHGS
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_ordered_crossover <src.models.policies.classical.hybrid_genetic_search.vectorized_ordered_crossover>`
  - ```{autodoc2-docstring} src.models.policies.classical.hybrid_genetic_search.vectorized_ordered_crossover
    :summary:
    ```
* - {py:obj}`calc_broken_pairs_distance <src.models.policies.classical.hybrid_genetic_search.calc_broken_pairs_distance>`
  - ```{autodoc2-docstring} src.models.policies.classical.hybrid_genetic_search.calc_broken_pairs_distance
    :summary:
    ```
````

### API

````{py:function} vectorized_ordered_crossover(parent1, parent2)
:canonical: src.models.policies.classical.hybrid_genetic_search.vectorized_ordered_crossover

```{autodoc2-docstring} src.models.policies.classical.hybrid_genetic_search.vectorized_ordered_crossover
```
````

````{py:function} calc_broken_pairs_distance(population)
:canonical: src.models.policies.classical.hybrid_genetic_search.calc_broken_pairs_distance

```{autodoc2-docstring} src.models.policies.classical.hybrid_genetic_search.calc_broken_pairs_distance
```
````

`````{py:class} VectorizedPopulation(size, device, alpha_diversity=0.5)
:canonical: src.models.policies.classical.hybrid_genetic_search.VectorizedPopulation

```{autodoc2-docstring} src.models.policies.classical.hybrid_genetic_search.VectorizedPopulation
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.classical.hybrid_genetic_search.VectorizedPopulation.__init__
```

````{py:method} initialize(initial_pop, initial_costs)
:canonical: src.models.policies.classical.hybrid_genetic_search.VectorizedPopulation.initialize

```{autodoc2-docstring} src.models.policies.classical.hybrid_genetic_search.VectorizedPopulation.initialize
```

````

````{py:method} add_individuals(candidates, costs)
:canonical: src.models.policies.classical.hybrid_genetic_search.VectorizedPopulation.add_individuals

```{autodoc2-docstring} src.models.policies.classical.hybrid_genetic_search.VectorizedPopulation.add_individuals
```

````

````{py:method} compute_biased_fitness()
:canonical: src.models.policies.classical.hybrid_genetic_search.VectorizedPopulation.compute_biased_fitness

```{autodoc2-docstring} src.models.policies.classical.hybrid_genetic_search.VectorizedPopulation.compute_biased_fitness
```

````

````{py:method} get_parents(n_offspring=1)
:canonical: src.models.policies.classical.hybrid_genetic_search.VectorizedPopulation.get_parents

```{autodoc2-docstring} src.models.policies.classical.hybrid_genetic_search.VectorizedPopulation.get_parents
```

````

`````

`````{py:class} VectorizedHGS(dist_matrix, demands, vehicle_capacity, time_limit=1.0, device='cuda')
:canonical: src.models.policies.classical.hybrid_genetic_search.VectorizedHGS

```{autodoc2-docstring} src.models.policies.classical.hybrid_genetic_search.VectorizedHGS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.classical.hybrid_genetic_search.VectorizedHGS.__init__
```

````{py:method} solve(initial_solutions, n_generations=50, population_size=10, elite_size=5, time_limit=None, max_vehicles=0)
:canonical: src.models.policies.classical.hybrid_genetic_search.VectorizedHGS.solve

```{autodoc2-docstring} src.models.policies.classical.hybrid_genetic_search.VectorizedHGS.solve
```

````

`````
