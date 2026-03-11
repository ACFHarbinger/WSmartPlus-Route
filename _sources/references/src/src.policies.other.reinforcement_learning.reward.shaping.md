# {py:mod}`src.policies.other.reinforcement_learning.reward.shaping`

```{py:module} src.policies.other.reinforcement_learning.reward.shaping
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.reward.shaping
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RewardShaper <src.policies.other.reinforcement_learning.reward.shaping.RewardShaper>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.reward.shaping.RewardShaper
    :summary:
    ```
* - {py:obj}`AdaptiveRewardShaper <src.policies.other.reinforcement_learning.reward.shaping.AdaptiveRewardShaper>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.reward.shaping.AdaptiveRewardShaper
    :summary:
    ```
````

### API

`````{py:class} RewardShaper(best_improvement_reward: float = 10.0, local_improvement_reward: float = 5.0, accepted_reward: float = 1.0, rejected_reward: float = -1.0, stagnation_penalty: float = -0.1)
:canonical: src.policies.other.reinforcement_learning.reward.shaping.RewardShaper

```{autodoc2-docstring} src.policies.other.reinforcement_learning.reward.shaping.RewardShaper
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.reward.shaping.RewardShaper.__init__
```

````{py:method} compute_reward(new_profit: float, prev_profit: float, best_profit: float, accepted: bool, stagnation_count: int = 0, improvement_threshold: float = 1e-06) -> float
:canonical: src.policies.other.reinforcement_learning.reward.shaping.RewardShaper.compute_reward

```{autodoc2-docstring} src.policies.other.reinforcement_learning.reward.shaping.RewardShaper.compute_reward
```

````

````{py:method} calculate_reward(new_cost: float, current_cost: float, best_cost: float, accepted: bool, stagnation_count: int = 0) -> float
:canonical: src.policies.other.reinforcement_learning.reward.shaping.RewardShaper.calculate_reward

```{autodoc2-docstring} src.policies.other.reinforcement_learning.reward.shaping.RewardShaper.calculate_reward
```

````

````{py:method} get_reward_config() -> typing.Dict[str, float]
:canonical: src.policies.other.reinforcement_learning.reward.shaping.RewardShaper.get_reward_config

```{autodoc2-docstring} src.policies.other.reinforcement_learning.reward.shaping.RewardShaper.get_reward_config
```

````

`````

`````{py:class} AdaptiveRewardShaper(*args, adaptation_rate: float = 0.5, **kwargs)
:canonical: src.policies.other.reinforcement_learning.reward.shaping.AdaptiveRewardShaper

Bases: {py:obj}`src.policies.other.reinforcement_learning.reward.shaping.RewardShaper`

```{autodoc2-docstring} src.policies.other.reinforcement_learning.reward.shaping.AdaptiveRewardShaper
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.reward.shaping.AdaptiveRewardShaper.__init__
```

````{py:method} adapt(progress: float)
:canonical: src.policies.other.reinforcement_learning.reward.shaping.AdaptiveRewardShaper.adapt

```{autodoc2-docstring} src.policies.other.reinforcement_learning.reward.shaping.AdaptiveRewardShaper.adapt
```

````

````{py:method} compute_reward(*args, progress: float = 0.5, **kwargs) -> float
:canonical: src.policies.other.reinforcement_learning.reward.shaping.AdaptiveRewardShaper.compute_reward

````

`````
