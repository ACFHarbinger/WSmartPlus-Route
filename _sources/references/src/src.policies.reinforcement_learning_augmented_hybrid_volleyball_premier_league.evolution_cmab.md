# {py:mod}`src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab`

```{py:module} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab
```

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CMABEvolution <src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution>`
  - ```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`update_biased_fitness <src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.update_biased_fitness>`
  - ```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.update_biased_fitness
    :summary:
    ```
````

### API

`````{py:class} CMABEvolution(split_manager: src.policies.hybrid_genetic_search.split.LinearSplit, bandit_algorithm: str = 'linucb', max_iterations: int = 1000, quality_weight: float = 0.5, improvement_weight: float = 0.6, diversity_weight: float = 0.2, novelty_weight: float = 1.0, reward_threshold: float = 1e-06, default_reward: float = 5.0, rng: typing.Optional[random.Random] = None, **kwargs)
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution.__init__
```

````{py:method} crossover(p1: src.policies.hybrid_genetic_search.individual.Individual, p2: src.policies.hybrid_genetic_search.individual.Individual, population: typing.List[src.policies.hybrid_genetic_search.individual.Individual], iteration: int, max_iterations: int) -> src.policies.hybrid_genetic_search.individual.Individual
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution.crossover

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution.crossover
```

````

````{py:method} _calculate_reward(child: src.policies.hybrid_genetic_search.individual.Individual, p1: src.policies.hybrid_genetic_search.individual.Individual, p2: src.policies.hybrid_genetic_search.individual.Individual, population: typing.List[src.policies.hybrid_genetic_search.individual.Individual]) -> float
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution._calculate_reward

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution._calculate_reward
```

````

````{py:method} evaluate(ind: src.policies.hybrid_genetic_search.individual.Individual)
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution.evaluate

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution.evaluate
```

````

````{py:method} update_improvement(improvement: float)
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution.update_improvement

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution.update_improvement
```

````

````{py:method} decay_exploration()
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution.decay_exploration

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution.decay_exploration
```

````

````{py:method} get_statistics() -> dict
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution.get_statistics

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.CMABEvolution.get_statistics
```

````

`````

````{py:function} update_biased_fitness(population: typing.List[src.policies.hybrid_genetic_search.individual.Individual], nb_elite: int, alpha_diversity: float = 0.5, neighbor_size: int = 15)
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.update_biased_fitness

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.evolution_cmab.update_biased_fitness
```
````
