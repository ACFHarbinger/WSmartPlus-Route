# {py:mod}`src.policies.hybrid_genetic_search.evolution`

```{py:module} src.policies.hybrid_genetic_search.evolution
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search.evolution
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ordered_crossover <src.policies.hybrid_genetic_search.evolution.ordered_crossover>`
  - ```{autodoc2-docstring} src.policies.hybrid_genetic_search.evolution.ordered_crossover
    :summary:
    ```
* - {py:obj}`update_biased_fitness <src.policies.hybrid_genetic_search.evolution.update_biased_fitness>`
  - ```{autodoc2-docstring} src.policies.hybrid_genetic_search.evolution.update_biased_fitness
    :summary:
    ```
* - {py:obj}`evaluate <src.policies.hybrid_genetic_search.evolution.evaluate>`
  - ```{autodoc2-docstring} src.policies.hybrid_genetic_search.evolution.evaluate
    :summary:
    ```
````

### API

````{py:function} ordered_crossover(p1: src.policies.hybrid_genetic_search.individual.Individual, p2: src.policies.hybrid_genetic_search.individual.Individual) -> src.policies.hybrid_genetic_search.individual.Individual
:canonical: src.policies.hybrid_genetic_search.evolution.ordered_crossover

```{autodoc2-docstring} src.policies.hybrid_genetic_search.evolution.ordered_crossover
```
````

````{py:function} update_biased_fitness(population: typing.List[src.policies.hybrid_genetic_search.individual.Individual], nb_elite: int)
:canonical: src.policies.hybrid_genetic_search.evolution.update_biased_fitness

```{autodoc2-docstring} src.policies.hybrid_genetic_search.evolution.update_biased_fitness
```
````

````{py:function} evaluate(ind: src.policies.hybrid_genetic_search.individual.Individual, split_manager: src.policies.hybrid_genetic_search.split.LinearSplit)
:canonical: src.policies.hybrid_genetic_search.evolution.evaluate

```{autodoc2-docstring} src.policies.hybrid_genetic_search.evolution.evaluate
```
````
