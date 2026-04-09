# {py:mod}`src.models.policies.hgs_core.population`

```{py:module} src.models.policies.hgs_core.population
```

```{autodoc2-docstring} src.models.policies.hgs_core.population
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedPopulation <src.models.policies.hgs_core.population.VectorizedPopulation>`
  - ```{autodoc2-docstring} src.models.policies.hgs_core.population.VectorizedPopulation
    :summary:
    ```
````

### API

`````{py:class} VectorizedPopulation(size: int, device: typing.Any, generator: typing.Optional[torch.Generator] = None)
:canonical: src.models.policies.hgs_core.population.VectorizedPopulation

```{autodoc2-docstring} src.models.policies.hgs_core.population.VectorizedPopulation
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.hgs_core.population.VectorizedPopulation.__init__
```

````{py:method} __getstate__()
:canonical: src.models.policies.hgs_core.population.VectorizedPopulation.__getstate__

```{autodoc2-docstring} src.models.policies.hgs_core.population.VectorizedPopulation.__getstate__
```

````

````{py:method} __setstate__(state)
:canonical: src.models.policies.hgs_core.population.VectorizedPopulation.__setstate__

```{autodoc2-docstring} src.models.policies.hgs_core.population.VectorizedPopulation.__setstate__
```

````

````{py:method} initialize(initial_pop: torch.Tensor, initial_costs: torch.Tensor, nb_elite: int)
:canonical: src.models.policies.hgs_core.population.VectorizedPopulation.initialize

```{autodoc2-docstring} src.models.policies.hgs_core.population.VectorizedPopulation.initialize
```

````

````{py:method} add_individuals(candidates: torch.Tensor, costs: torch.Tensor, nb_elite: int)
:canonical: src.models.policies.hgs_core.population.VectorizedPopulation.add_individuals

```{autodoc2-docstring} src.models.policies.hgs_core.population.VectorizedPopulation.add_individuals
```

````

````{py:method} compute_biased_fitness(nb_elite: int)
:canonical: src.models.policies.hgs_core.population.VectorizedPopulation.compute_biased_fitness

```{autodoc2-docstring} src.models.policies.hgs_core.population.VectorizedPopulation.compute_biased_fitness
```

````

````{py:method} get_parents(n_offspring: int = 1) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.policies.hgs_core.population.VectorizedPopulation.get_parents

```{autodoc2-docstring} src.models.policies.hgs_core.population.VectorizedPopulation.get_parents
```

````

`````
