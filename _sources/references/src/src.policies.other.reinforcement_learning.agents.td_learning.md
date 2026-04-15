# {py:mod}`src.policies.other.reinforcement_learning.agents.td_learning`

```{py:module} src.policies.other.reinforcement_learning.agents.td_learning
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TDAgent <src.policies.other.reinforcement_learning.agents.td_learning.TDAgent>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.TDAgent
    :summary:
    ```
* - {py:obj}`QLearningAgent <src.policies.other.reinforcement_learning.agents.td_learning.QLearningAgent>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.QLearningAgent
    :summary:
    ```
* - {py:obj}`SarsaAgent <src.policies.other.reinforcement_learning.agents.td_learning.SarsaAgent>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.SarsaAgent
    :summary:
    ```
* - {py:obj}`ExpectedSarsaAgent <src.policies.other.reinforcement_learning.agents.td_learning.ExpectedSarsaAgent>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.ExpectedSarsaAgent
    :summary:
    ```
````

### API

`````{py:class} TDAgent(n_states: int, n_actions: int, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1, epsilon_decay: float = 0.995, epsilon_min: float = 0.05, history_size: int = 100, seed: int = 42)
:canonical: src.policies.other.reinforcement_learning.agents.td_learning.TDAgent

Bases: {py:obj}`src.policies.other.reinforcement_learning.agents.base.RLAgent`

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.TDAgent
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.TDAgent.__init__
```

````{py:method} get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.other.reinforcement_learning.agents.td_learning.TDAgent.get_statistics

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.TDAgent.get_statistics
```

````

````{py:method} get_q_values(state: typing.Any) -> numpy.ndarray
:canonical: src.policies.other.reinforcement_learning.agents.td_learning.TDAgent.get_q_values

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.TDAgent.get_q_values
```

````

````{py:method} select_action(state: typing.Any, rng: numpy.random.Generator) -> int
:canonical: src.policies.other.reinforcement_learning.agents.td_learning.TDAgent.select_action

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.TDAgent.select_action
```

````

````{py:method} decay_epsilon() -> None
:canonical: src.policies.other.reinforcement_learning.agents.td_learning.TDAgent.decay_epsilon

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.TDAgent.decay_epsilon
```

````

````{py:method} save(path: str) -> None
:canonical: src.policies.other.reinforcement_learning.agents.td_learning.TDAgent.save

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.TDAgent.save
```

````

````{py:method} load(path: str) -> None
:canonical: src.policies.other.reinforcement_learning.agents.td_learning.TDAgent.load

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.TDAgent.load
```

````

````{py:method} reset() -> None
:canonical: src.policies.other.reinforcement_learning.agents.td_learning.TDAgent.reset

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.TDAgent.reset
```

````

`````

`````{py:class} QLearningAgent(n_states: int, n_actions: int, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1, epsilon_decay: float = 0.995, epsilon_min: float = 0.05, history_size: int = 100, seed: int = 42)
:canonical: src.policies.other.reinforcement_learning.agents.td_learning.QLearningAgent

Bases: {py:obj}`src.policies.other.reinforcement_learning.agents.td_learning.TDAgent`

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.QLearningAgent
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.QLearningAgent.__init__
```

````{py:method} update(state: typing.Any, action: int, reward: float, next_state: typing.Any, done: bool) -> None
:canonical: src.policies.other.reinforcement_learning.agents.td_learning.QLearningAgent.update

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.QLearningAgent.update
```

````

`````

`````{py:class} SarsaAgent(*args: typing.Any, **kwargs: typing.Any)
:canonical: src.policies.other.reinforcement_learning.agents.td_learning.SarsaAgent

Bases: {py:obj}`src.policies.other.reinforcement_learning.agents.td_learning.TDAgent`

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.SarsaAgent
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.SarsaAgent.__init__
```

````{py:method} update(state: typing.Any, action: int, reward: float, next_state: typing.Any, done: bool, next_action: typing.Optional[int] = None) -> None
:canonical: src.policies.other.reinforcement_learning.agents.td_learning.SarsaAgent.update

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.SarsaAgent.update
```

````

`````

`````{py:class} ExpectedSarsaAgent(n_states: int, n_actions: int, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1, epsilon_decay: float = 0.995, epsilon_min: float = 0.05, history_size: int = 100, seed: int = 42)
:canonical: src.policies.other.reinforcement_learning.agents.td_learning.ExpectedSarsaAgent

Bases: {py:obj}`src.policies.other.reinforcement_learning.agents.td_learning.TDAgent`

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.ExpectedSarsaAgent
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.ExpectedSarsaAgent.__init__
```

````{py:method} update(state: typing.Any, action: int, reward: float, next_state: typing.Any, done: bool) -> None
:canonical: src.policies.other.reinforcement_learning.agents.td_learning.ExpectedSarsaAgent.update

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.td_learning.ExpectedSarsaAgent.update
```

````

`````
