# {py:mod}`src.policies.other.reinforcement_learning.agents.bandits.epsilon_greedy`

```{py:module} src.policies.other.reinforcement_learning.agents.bandits.epsilon_greedy
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.epsilon_greedy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EpsilonGreedyBandit <src.policies.other.reinforcement_learning.agents.bandits.epsilon_greedy.EpsilonGreedyBandit>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.epsilon_greedy.EpsilonGreedyBandit
    :summary:
    ```
````

### API

`````{py:class} EpsilonGreedyBandit(n_arms: int, epsilon: float = 0.1, epsilon_decay: float = 0.999, epsilon_min: float = 0.01, seed: typing.Optional[int] = None, history_size: int = 50)
:canonical: src.policies.other.reinforcement_learning.agents.bandits.epsilon_greedy.EpsilonGreedyBandit

Bases: {py:obj}`src.policies.other.reinforcement_learning.agents.bandits.base.BanditAgent`

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.epsilon_greedy.EpsilonGreedyBandit
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.epsilon_greedy.EpsilonGreedyBandit.__init__
```

````{py:method} decay_epsilon() -> None
:canonical: src.policies.other.reinforcement_learning.agents.bandits.epsilon_greedy.EpsilonGreedyBandit.decay_epsilon

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.epsilon_greedy.EpsilonGreedyBandit.decay_epsilon
```

````

````{py:method} select_action(state: typing.Any, rng: typing.Optional[numpy.random.Generator] = None) -> int
:canonical: src.policies.other.reinforcement_learning.agents.bandits.epsilon_greedy.EpsilonGreedyBandit.select_action

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.epsilon_greedy.EpsilonGreedyBandit.select_action
```

````

````{py:method} get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.other.reinforcement_learning.agents.bandits.epsilon_greedy.EpsilonGreedyBandit.get_statistics

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.epsilon_greedy.EpsilonGreedyBandit.get_statistics
```

````

`````
