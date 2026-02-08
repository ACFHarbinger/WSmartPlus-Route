# {py:mod}`src.models.policies.hgs.population`

```{py:module} src.models.policies.hgs.population
```

```{autodoc2-docstring} src.models.policies.hgs.population
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedPopulation <src.models.policies.hgs.population.VectorizedPopulation>`
  - ```{autodoc2-docstring} src.models.policies.hgs.population.VectorizedPopulation
    :summary:
    ```
````

### API

`````{py:class} VectorizedPopulation(size: int, device: typing.Any, alpha_diversity: float = 0.5)
:canonical: src.models.policies.hgs.population.VectorizedPopulation

```{autodoc2-docstring} src.models.policies.hgs.population.VectorizedPopulation
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.hgs.population.VectorizedPopulation.__init__
```

````{py:method} initialize(initial_pop: torch.Tensor, initial_costs: torch.Tensor)
:canonical: src.models.policies.hgs.population.VectorizedPopulation.initialize

```{autodoc2-docstring} src.models.policies.hgs.population.VectorizedPopulation.initialize
```

````

````{py:method} add_individuals(candidates: torch.Tensor, costs: torch.Tensor)
:canonical: src.models.policies.hgs.population.VectorizedPopulation.add_individuals

```{autodoc2-docstring} src.models.policies.hgs.population.VectorizedPopulation.add_individuals
```

````

````{py:method} compute_biased_fitness()
:canonical: src.models.policies.hgs.population.VectorizedPopulation.compute_biased_fitness

```{autodoc2-docstring} src.models.policies.hgs.population.VectorizedPopulation.compute_biased_fitness
```

````

````{py:method} get_parents(n_offspring: int = 1) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.policies.hgs.population.VectorizedPopulation.get_parents

```{autodoc2-docstring} src.models.policies.hgs.population.VectorizedPopulation.get_parents
```

````

`````
