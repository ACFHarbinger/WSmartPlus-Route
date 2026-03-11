# {py:mod}`src.policies.other.reinforcement_learning.agents.bandits.exp3`

```{py:module} src.policies.other.reinforcement_learning.agents.bandits.exp3
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.exp3
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EXP3Agent <src.policies.other.reinforcement_learning.agents.bandits.exp3.EXP3Agent>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.exp3.EXP3Agent
    :summary:
    ```
````

### API

`````{py:class} EXP3Agent(n_arms: int, gamma: float = 0.1, seed: typing.Optional[int] = None, history_size: int = 50)
:canonical: src.policies.other.reinforcement_learning.agents.bandits.exp3.EXP3Agent

Bases: {py:obj}`src.policies.other.reinforcement_learning.agents.bandits.base.BanditAgent`

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.exp3.EXP3Agent
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.exp3.EXP3Agent.__init__
```

````{py:method} select_action(state: typing.Any, rng: typing.Optional[numpy.random.Generator] = None) -> int
:canonical: src.policies.other.reinforcement_learning.agents.bandits.exp3.EXP3Agent.select_action

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.exp3.EXP3Agent.select_action
```

````

````{py:method} update(state: typing.Any, action: int, reward: float, next_state: typing.Any, done: bool) -> None
:canonical: src.policies.other.reinforcement_learning.agents.bandits.exp3.EXP3Agent.update

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.exp3.EXP3Agent.update
```

````

````{py:method} reset() -> None
:canonical: src.policies.other.reinforcement_learning.agents.bandits.exp3.EXP3Agent.reset

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.exp3.EXP3Agent.reset
```

````

````{py:method} get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.other.reinforcement_learning.agents.bandits.exp3.EXP3Agent.get_statistics

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.exp3.EXP3Agent.get_statistics
```

````

`````
