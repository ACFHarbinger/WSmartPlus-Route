# {py:mod}`src.policies.other.reinforcement_learning.features.context`

```{py:module} src.policies.other.reinforcement_learning.features.context
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.features.context
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ContextFeatureExtractor <src.policies.other.reinforcement_learning.features.context.ContextFeatureExtractor>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.features.context.ContextFeatureExtractor
    :summary:
    ```
````

### API

`````{py:class} ContextFeatureExtractor(diversity_history_size: int = 10, improvement_history_size: int = 10)
:canonical: src.policies.other.reinforcement_learning.features.context.ContextFeatureExtractor

```{autodoc2-docstring} src.policies.other.reinforcement_learning.features.context.ContextFeatureExtractor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.features.context.ContextFeatureExtractor.__init__
```

````{py:method} extract_features(p1: src.policies.hybrid_genetic_search.individual.Individual, p2: src.policies.hybrid_genetic_search.individual.Individual, population: typing.List[src.policies.hybrid_genetic_search.individual.Individual], iteration: int, progress: float = 0.0) -> numpy.ndarray
:canonical: src.policies.other.reinforcement_learning.features.context.ContextFeatureExtractor.extract_features

```{autodoc2-docstring} src.policies.other.reinforcement_learning.features.context.ContextFeatureExtractor.extract_features
```

````

````{py:method} _calculate_population_diversity(population: typing.List[src.policies.hybrid_genetic_search.individual.Individual]) -> float
:canonical: src.policies.other.reinforcement_learning.features.context.ContextFeatureExtractor._calculate_population_diversity

```{autodoc2-docstring} src.policies.other.reinforcement_learning.features.context.ContextFeatureExtractor._calculate_population_diversity
```

````

````{py:method} _individual_diversity(ind: src.policies.hybrid_genetic_search.individual.Individual, population: typing.List[src.policies.hybrid_genetic_search.individual.Individual]) -> float
:canonical: src.policies.other.reinforcement_learning.features.context.ContextFeatureExtractor._individual_diversity

```{autodoc2-docstring} src.policies.other.reinforcement_learning.features.context.ContextFeatureExtractor._individual_diversity
```

````

````{py:method} update_improvement(improvement: float)
:canonical: src.policies.other.reinforcement_learning.features.context.ContextFeatureExtractor.update_improvement

```{autodoc2-docstring} src.policies.other.reinforcement_learning.features.context.ContextFeatureExtractor.update_improvement
```

````

`````
