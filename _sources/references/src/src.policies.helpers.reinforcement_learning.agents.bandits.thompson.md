# {py:mod}`src.policies.helpers.reinforcement_learning.agents.bandits.thompson`

```{py:module} src.policies.helpers.reinforcement_learning.agents.bandits.thompson
```

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.thompson
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ThompsonSamplingBandit <src.policies.helpers.reinforcement_learning.agents.bandits.thompson.ThompsonSamplingBandit>`
  - ```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.thompson.ThompsonSamplingBandit
    :summary:
    ```
````

### API

`````{py:class} ThompsonSamplingBandit(n_arms: int, alpha_prior: float = 1.0, beta_prior: float = 1.0, seed: typing.Optional[int] = None, history_size: int = 50)
:canonical: src.policies.helpers.reinforcement_learning.agents.bandits.thompson.ThompsonSamplingBandit

Bases: {py:obj}`src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent`

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.thompson.ThompsonSamplingBandit
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.thompson.ThompsonSamplingBandit.__init__
```

````{py:method} select_action(state: typing.Any, rng: numpy.random.Generator) -> int
:canonical: src.policies.helpers.reinforcement_learning.agents.bandits.thompson.ThompsonSamplingBandit.select_action

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.thompson.ThompsonSamplingBandit.select_action
```

````

````{py:method} update(state: typing.Any, action: int, reward: float, next_state: typing.Any, done: bool) -> None
:canonical: src.policies.helpers.reinforcement_learning.agents.bandits.thompson.ThompsonSamplingBandit.update

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.thompson.ThompsonSamplingBandit.update
```

````

````{py:method} reset() -> None
:canonical: src.policies.helpers.reinforcement_learning.agents.bandits.thompson.ThompsonSamplingBandit.reset

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.thompson.ThompsonSamplingBandit.reset
```

````

````{py:method} get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.helpers.reinforcement_learning.agents.bandits.thompson.ThompsonSamplingBandit.get_statistics

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.thompson.ThompsonSamplingBandit.get_statistics
```

````

`````
