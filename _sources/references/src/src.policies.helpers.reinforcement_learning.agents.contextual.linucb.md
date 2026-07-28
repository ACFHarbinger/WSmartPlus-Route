# {py:mod}`src.policies.helpers.reinforcement_learning.agents.contextual.linucb`

```{py:module} src.policies.helpers.reinforcement_learning.agents.contextual.linucb
```

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.linucb
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LinUCBAgent <src.policies.helpers.reinforcement_learning.agents.contextual.linucb.LinUCBAgent>`
  - ```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.linucb.LinUCBAgent
    :summary:
    ```
````

### API

`````{py:class} LinUCBAgent(n_arms: int, feature_dim: int, alpha: float = 1.0, seed: typing.Optional[int] = None, history_size: int = 50)
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.linucb.LinUCBAgent

Bases: {py:obj}`src.policies.helpers.reinforcement_learning.agents.contextual.base.ContextualBanditAgent`

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.linucb.LinUCBAgent
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.linucb.LinUCBAgent.__init__
```

````{py:method} select_action(context: numpy.ndarray, rng: numpy.random.Generator) -> int
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.linucb.LinUCBAgent.select_action

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.linucb.LinUCBAgent.select_action
```

````

````{py:method} update(context: numpy.ndarray, action: int, reward: float, next_context: typing.Any = None, done: bool = False) -> None
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.linucb.LinUCBAgent.update

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.linucb.LinUCBAgent.update
```

````

````{py:method} get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.linucb.LinUCBAgent.get_statistics

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.linucb.LinUCBAgent.get_statistics
```

````

````{py:method} get_weights() -> typing.List[numpy.ndarray]
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.linucb.LinUCBAgent.get_weights

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.linucb.LinUCBAgent.get_weights
```

````

`````
