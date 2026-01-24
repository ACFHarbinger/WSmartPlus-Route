# {py:mod}`src.policies.hgs_aux.evolution`

```{py:module} src.policies.hgs_aux.evolution
```

```{autodoc2-docstring} src.policies.hgs_aux.evolution
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ordered_crossover <src.policies.hgs_aux.evolution.ordered_crossover>`
  - ```{autodoc2-docstring} src.policies.hgs_aux.evolution.ordered_crossover
    :summary:
    ```
* - {py:obj}`update_biased_fitness <src.policies.hgs_aux.evolution.update_biased_fitness>`
  - ```{autodoc2-docstring} src.policies.hgs_aux.evolution.update_biased_fitness
    :summary:
    ```
* - {py:obj}`evaluate <src.policies.hgs_aux.evolution.evaluate>`
  - ```{autodoc2-docstring} src.policies.hgs_aux.evolution.evaluate
    :summary:
    ```
````

### API

````{py:function} ordered_crossover(p1: src.policies.hgs_aux.types.Individual, p2: src.policies.hgs_aux.types.Individual) -> src.policies.hgs_aux.types.Individual
:canonical: src.policies.hgs_aux.evolution.ordered_crossover

```{autodoc2-docstring} src.policies.hgs_aux.evolution.ordered_crossover
```
````

````{py:function} update_biased_fitness(population: typing.List[src.policies.hgs_aux.types.Individual], nb_elite: int)
:canonical: src.policies.hgs_aux.evolution.update_biased_fitness

```{autodoc2-docstring} src.policies.hgs_aux.evolution.update_biased_fitness
```
````

````{py:function} evaluate(ind: src.policies.hgs_aux.types.Individual, split_manager)
:canonical: src.policies.hgs_aux.evolution.evaluate

```{autodoc2-docstring} src.policies.hgs_aux.evolution.evaluate
```
````
