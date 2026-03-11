# {py:mod}`src.policies.other.reinforcement_learning.agents.contextual.base`

```{py:module} src.policies.other.reinforcement_learning.agents.contextual.base
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.contextual.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ContextualBanditAgent <src.policies.other.reinforcement_learning.agents.contextual.base.ContextualBanditAgent>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.contextual.base.ContextualBanditAgent
    :summary:
    ```
````

### API

`````{py:class} ContextualBanditAgent(n_arms: int, feature_dim: int, seed: typing.Optional[int] = None, history_size: int = 50)
:canonical: src.policies.other.reinforcement_learning.agents.contextual.base.ContextualBanditAgent

Bases: {py:obj}`src.policies.other.reinforcement_learning.agents.base.RLAgent`

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.contextual.base.ContextualBanditAgent
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.contextual.base.ContextualBanditAgent.__init__
```

````{py:method} decay_epsilon()
:canonical: src.policies.other.reinforcement_learning.agents.contextual.base.ContextualBanditAgent.decay_epsilon

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.contextual.base.ContextualBanditAgent.decay_epsilon
```

````

````{py:method} select_action(context: numpy.ndarray, rng: numpy.random.Generator) -> int
:canonical: src.policies.other.reinforcement_learning.agents.contextual.base.ContextualBanditAgent.select_action
:abstractmethod:

````

````{py:method} get_weights() -> typing.Any
:canonical: src.policies.other.reinforcement_learning.agents.contextual.base.ContextualBanditAgent.get_weights

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.contextual.base.ContextualBanditAgent.get_weights
```

````

````{py:method} save(path: str) -> None
:canonical: src.policies.other.reinforcement_learning.agents.contextual.base.ContextualBanditAgent.save

````

````{py:method} load(path: str) -> None
:canonical: src.policies.other.reinforcement_learning.agents.contextual.base.ContextualBanditAgent.load

````

````{py:method} reset() -> None
:canonical: src.policies.other.reinforcement_learning.agents.contextual.base.ContextualBanditAgent.reset

````

`````
