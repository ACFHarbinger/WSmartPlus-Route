# {py:mod}`src.policies.other.reinforcement_learning.agents.base`

```{py:module} src.policies.other.reinforcement_learning.agents.base
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLAgent <src.policies.other.reinforcement_learning.agents.base.RLAgent>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.base.RLAgent
    :summary:
    ```
````

### API

`````{py:class} RLAgent
:canonical: src.policies.other.reinforcement_learning.agents.base.RLAgent

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.base.RLAgent
```

````{py:method} select_action(state: typing.Any, rng: numpy.random.Generator) -> int
:canonical: src.policies.other.reinforcement_learning.agents.base.RLAgent.select_action
:abstractmethod:

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.base.RLAgent.select_action
```

````

````{py:method} update(state: typing.Any, action: int, reward: float, next_state: typing.Any, done: bool) -> None
:canonical: src.policies.other.reinforcement_learning.agents.base.RLAgent.update
:abstractmethod:

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.base.RLAgent.update
```

````

````{py:method} save(path: str) -> None
:canonical: src.policies.other.reinforcement_learning.agents.base.RLAgent.save
:abstractmethod:

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.base.RLAgent.save
```

````

````{py:method} load(path: str) -> None
:canonical: src.policies.other.reinforcement_learning.agents.base.RLAgent.load
:abstractmethod:

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.base.RLAgent.load
```

````

````{py:method} reset() -> None
:canonical: src.policies.other.reinforcement_learning.agents.base.RLAgent.reset
:abstractmethod:

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.base.RLAgent.reset
```

````

````{py:method} get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.other.reinforcement_learning.agents.base.RLAgent.get_statistics

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.base.RLAgent.get_statistics
```

````

`````
