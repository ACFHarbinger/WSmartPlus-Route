# {py:mod}`src.policies.acceptance_criteria.adaptive_boltzmann_metropolis`

```{py:module} src.policies.acceptance_criteria.adaptive_boltzmann_metropolis
```

```{autodoc2-docstring} src.policies.acceptance_criteria.adaptive_boltzmann_metropolis
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AdaptiveBoltzmannMetropolis <src.policies.acceptance_criteria.adaptive_boltzmann_metropolis.AdaptiveBoltzmannMetropolis>`
  - ```{autodoc2-docstring} src.policies.acceptance_criteria.adaptive_boltzmann_metropolis.AdaptiveBoltzmannMetropolis
    :summary:
    ```
````

### API

`````{py:class} AdaptiveBoltzmannMetropolis(p0: float = 0.5, window_size: int = 100, alpha: float = 0.95, min_temp: float = 1e-06, seed: typing.Optional[int] = None, maximization: bool = True)
:canonical: src.policies.acceptance_criteria.adaptive_boltzmann_metropolis.AdaptiveBoltzmannMetropolis

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.acceptance_criteria.adaptive_boltzmann_metropolis.AdaptiveBoltzmannMetropolis
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.acceptance_criteria.adaptive_boltzmann_metropolis.AdaptiveBoltzmannMetropolis.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.acceptance_criteria.adaptive_boltzmann_metropolis.AdaptiveBoltzmannMetropolis.setup

```{autodoc2-docstring} src.policies.acceptance_criteria.adaptive_boltzmann_metropolis.AdaptiveBoltzmannMetropolis.setup
```

````

````{py:method} _update_stats(delta: float) -> None
:canonical: src.policies.acceptance_criteria.adaptive_boltzmann_metropolis.AdaptiveBoltzmannMetropolis._update_stats

```{autodoc2-docstring} src.policies.acceptance_criteria.adaptive_boltzmann_metropolis.AdaptiveBoltzmannMetropolis._update_stats
```

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.interfaces.context.search_context.AcceptanceMetrics]
:canonical: src.policies.acceptance_criteria.adaptive_boltzmann_metropolis.AdaptiveBoltzmannMetropolis.accept

```{autodoc2-docstring} src.policies.acceptance_criteria.adaptive_boltzmann_metropolis.AdaptiveBoltzmannMetropolis.accept
```

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.acceptance_criteria.adaptive_boltzmann_metropolis.AdaptiveBoltzmannMetropolis.step

```{autodoc2-docstring} src.policies.acceptance_criteria.adaptive_boltzmann_metropolis.AdaptiveBoltzmannMetropolis.step
```

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.acceptance_criteria.adaptive_boltzmann_metropolis.AdaptiveBoltzmannMetropolis.get_state

```{autodoc2-docstring} src.policies.acceptance_criteria.adaptive_boltzmann_metropolis.AdaptiveBoltzmannMetropolis.get_state
```

````

`````
