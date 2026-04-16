# {py:mod}`src.policies.helpers.reinforcement_learning.agents.bandits.base`

```{py:module} src.policies.helpers.reinforcement_learning.agents.bandits.base
```

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BanditAgent <src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent>`
  - ```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent
    :summary:
    ```
````

### API

`````{py:class} BanditAgent(n_arms: int, seed: typing.Optional[int] = None, history_size: int = 50)
:canonical: src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent

Bases: {py:obj}`src.policies.helpers.reinforcement_learning.agents.base.RLAgent`

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent.__init__
```

````{py:method} decay_epsilon() -> None
:canonical: src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent.decay_epsilon

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent.decay_epsilon
```

````

````{py:method} reset() -> None
:canonical: src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent.reset

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent.reset
```

````

````{py:method} save(path: str) -> None
:canonical: src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent.save

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent.save
```

````

````{py:method} load(path: str) -> None
:canonical: src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent.load

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent.load
```

````

````{py:method} get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent.get_statistics

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent.get_statistics
```

````

````{py:method} update(state: typing.Any, action: int, reward: float, next_state: typing.Any, done: bool) -> None
:canonical: src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent.update

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent.update
```

````

`````
