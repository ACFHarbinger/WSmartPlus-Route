# {py:mod}`src.policies.other.reinforcement_learning.agents.bandits.ucb`

```{py:module} src.policies.other.reinforcement_learning.agents.bandits.ucb
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`UCBBandit <src.policies.other.reinforcement_learning.agents.bandits.ucb.UCBBandit>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.UCBBandit
    :summary:
    ```
* - {py:obj}`DiscountedUCBBandit <src.policies.other.reinforcement_learning.agents.bandits.ucb.DiscountedUCBBandit>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.DiscountedUCBBandit
    :summary:
    ```
* - {py:obj}`SlidingWindowUCBBandit <src.policies.other.reinforcement_learning.agents.bandits.ucb.SlidingWindowUCBBandit>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.SlidingWindowUCBBandit
    :summary:
    ```
````

### API

`````{py:class} UCBBandit(n_arms: int, c: float = 2.0, seed: typing.Optional[int] = None, history_size: int = 50)
:canonical: src.policies.other.reinforcement_learning.agents.bandits.ucb.UCBBandit

Bases: {py:obj}`src.policies.other.reinforcement_learning.agents.bandits.base.BanditAgent`

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.UCBBandit
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.UCBBandit.__init__
```

````{py:method} select_action(state: typing.Any, rng: typing.Optional[numpy.random.Generator] = None) -> int
:canonical: src.policies.other.reinforcement_learning.agents.bandits.ucb.UCBBandit.select_action

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.UCBBandit.select_action
```

````

````{py:method} get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.other.reinforcement_learning.agents.bandits.ucb.UCBBandit.get_statistics

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.UCBBandit.get_statistics
```

````

`````

`````{py:class} DiscountedUCBBandit(n_arms: int, c: float = 2.0, gamma: float = 0.95, seed: typing.Optional[int] = None, history_size: int = 50)
:canonical: src.policies.other.reinforcement_learning.agents.bandits.ucb.DiscountedUCBBandit

Bases: {py:obj}`src.policies.other.reinforcement_learning.agents.bandits.base.BanditAgent`

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.DiscountedUCBBandit
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.DiscountedUCBBandit.__init__
```

````{py:method} select_action(state: typing.Any, rng: typing.Optional[numpy.random.Generator] = None) -> int
:canonical: src.policies.other.reinforcement_learning.agents.bandits.ucb.DiscountedUCBBandit.select_action

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.DiscountedUCBBandit.select_action
```

````

````{py:method} update(state: typing.Any, action: int, reward: float, next_state: typing.Any, done: bool) -> None
:canonical: src.policies.other.reinforcement_learning.agents.bandits.ucb.DiscountedUCBBandit.update

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.DiscountedUCBBandit.update
```

````

`````

`````{py:class} SlidingWindowUCBBandit(n_arms: int, window_size: int = 100, c: float = 2.0, seed: typing.Optional[int] = None, history_size: int = 50)
:canonical: src.policies.other.reinforcement_learning.agents.bandits.ucb.SlidingWindowUCBBandit

Bases: {py:obj}`src.policies.other.reinforcement_learning.agents.bandits.base.BanditAgent`

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.SlidingWindowUCBBandit
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.SlidingWindowUCBBandit.__init__
```

````{py:method} select_action(state: typing.Any, rng: typing.Optional[numpy.random.Generator] = None) -> int
:canonical: src.policies.other.reinforcement_learning.agents.bandits.ucb.SlidingWindowUCBBandit.select_action

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.SlidingWindowUCBBandit.select_action
```

````

````{py:method} reset() -> None
:canonical: src.policies.other.reinforcement_learning.agents.bandits.ucb.SlidingWindowUCBBandit.reset

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.SlidingWindowUCBBandit.reset
```

````

````{py:method} update(state: typing.Any, action: int, reward: float, next_state: typing.Any, done: bool) -> None
:canonical: src.policies.other.reinforcement_learning.agents.bandits.ucb.SlidingWindowUCBBandit.update

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.SlidingWindowUCBBandit.update
```

````

````{py:method} get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.other.reinforcement_learning.agents.bandits.ucb.SlidingWindowUCBBandit.get_statistics

```{autodoc2-docstring} src.policies.other.reinforcement_learning.agents.bandits.ucb.SlidingWindowUCBBandit.get_statistics
```

````

`````
