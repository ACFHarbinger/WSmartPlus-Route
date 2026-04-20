# {py:mod}`src.policies.helpers.reinforcement_learning.agents.bandits.softmax`

```{py:module} src.policies.helpers.reinforcement_learning.agents.bandits.softmax
```

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.softmax
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SoftmaxBandit <src.policies.helpers.reinforcement_learning.agents.bandits.softmax.SoftmaxBandit>`
  - ```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.softmax.SoftmaxBandit
    :summary:
    ```
````

### API

`````{py:class} SoftmaxBandit(n_arms: int, temperature: float = 1.0, seed: typing.Optional[int] = None, history_size: int = 50)
:canonical: src.policies.helpers.reinforcement_learning.agents.bandits.softmax.SoftmaxBandit

Bases: {py:obj}`src.policies.helpers.reinforcement_learning.agents.bandits.base.BanditAgent`

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.softmax.SoftmaxBandit
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.softmax.SoftmaxBandit.__init__
```

````{py:method} select_action(state: typing.Any, rng: numpy.random.Generator) -> int
:canonical: src.policies.helpers.reinforcement_learning.agents.bandits.softmax.SoftmaxBandit.select_action

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.softmax.SoftmaxBandit.select_action
```

````

````{py:method} get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.helpers.reinforcement_learning.agents.bandits.softmax.SoftmaxBandit.get_statistics

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.bandits.softmax.SoftmaxBandit.get_statistics
```

````

`````
