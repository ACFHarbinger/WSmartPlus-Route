# {py:mod}`src.policies.helpers.reinforcement_learning.agents.contextual.thompson`

```{py:module} src.policies.helpers.reinforcement_learning.agents.contextual.thompson
```

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.thompson
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ContextualThompsonSamplingAgent <src.policies.helpers.reinforcement_learning.agents.contextual.thompson.ContextualThompsonSamplingAgent>`
  - ```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.thompson.ContextualThompsonSamplingAgent
    :summary:
    ```
````

### API

`````{py:class} ContextualThompsonSamplingAgent(n_arms: int, feature_dim: int, lambda_prior: float = 1.0, noise_variance: float = 0.1, seed: typing.Optional[int] = None, history_size: int = 50)
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.thompson.ContextualThompsonSamplingAgent

Bases: {py:obj}`src.policies.helpers.reinforcement_learning.agents.contextual.base.ContextualBanditAgent`

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.thompson.ContextualThompsonSamplingAgent
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.thompson.ContextualThompsonSamplingAgent.__init__
```

````{py:method} select_action(context: numpy.ndarray, rng: numpy.random.Generator) -> int
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.thompson.ContextualThompsonSamplingAgent.select_action

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.thompson.ContextualThompsonSamplingAgent.select_action
```

````

````{py:method} update(context: numpy.ndarray, action: int, reward: float, next_context: typing.Any = None, done: bool = False) -> None
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.thompson.ContextualThompsonSamplingAgent.update

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.thompson.ContextualThompsonSamplingAgent.update
```

````

````{py:method} reset() -> None
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.thompson.ContextualThompsonSamplingAgent.reset

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.thompson.ContextualThompsonSamplingAgent.reset
```

````

````{py:method} get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.thompson.ContextualThompsonSamplingAgent.get_statistics

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.thompson.ContextualThompsonSamplingAgent.get_statistics
```

````

````{py:method} get_weights() -> typing.List[numpy.ndarray]
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.thompson.ContextualThompsonSamplingAgent.get_weights

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.thompson.ContextualThompsonSamplingAgent.get_weights
```

````

`````
