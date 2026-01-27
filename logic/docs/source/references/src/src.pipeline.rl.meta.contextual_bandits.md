# {py:mod}`src.pipeline.rl.meta.contextual_bandits`

```{py:module} src.pipeline.rl.meta.contextual_bandits
```

```{autodoc2-docstring} src.pipeline.rl.meta.contextual_bandits
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WeightContextualBandit <src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit>`
  - ```{autodoc2-docstring} src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit
    :summary:
    ```
````

### API

`````{py:class} WeightContextualBandit(num_days: int = 10, initial_weights: typing.Optional[typing.Dict[str, float]] = None, context_features: typing.Optional[typing.List[str]] = None, features_aggregation: str = 'avg', exploration_strategy: str = 'ucb', exploration_factor: float = 0.5, num_weight_configs: int = 10, weight_ranges: typing.Optional[typing.Dict[str, typing.Tuple[float, float]]] = None, window_size: int = 20, **kwargs)
:canonical: src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit

Bases: {py:obj}`logic.src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy`

```{autodoc2-docstring} src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit.__init__
```

````{py:method} _get_context_features(dataset)
:canonical: src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit._get_context_features

```{autodoc2-docstring} src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit._get_context_features
```

````

````{py:method} set_max_feature_values(max_vals)
:canonical: src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit.set_max_feature_values

```{autodoc2-docstring} src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit.set_max_feature_values
```

````

````{py:method} update(reward, metrics, context=None)
:canonical: src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit.update

```{autodoc2-docstring} src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit.update
```

````

````{py:method} propose_weights(context: typing.Optional[typing.Dict[str, typing.Any]] = None) -> typing.Dict[str, float]
:canonical: src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit.propose_weights

```{autodoc2-docstring} src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit.propose_weights
```

````

````{py:method} feedback(reward: float, metrics: typing.Any, day: typing.Optional[int] = None, step: typing.Optional[int] = None)
:canonical: src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit.feedback

```{autodoc2-docstring} src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit.feedback
```

````

````{py:method} get_current_weights(dataset=None) -> typing.Dict[str, float]
:canonical: src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit.get_current_weights

```{autodoc2-docstring} src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit.get_current_weights
```

````

````{py:method} _generate_weight_configs(initial_weights, num_configs)
:canonical: src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit._generate_weight_configs

```{autodoc2-docstring} src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit._generate_weight_configs
```

````

````{py:method} _context_to_key(context: typing.Dict[str, typing.Any]) -> typing.Tuple
:canonical: src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit._context_to_key

```{autodoc2-docstring} src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit._context_to_key
```

````

````{py:method} _select_ucb(context_key)
:canonical: src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit._select_ucb

```{autodoc2-docstring} src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit._select_ucb
```

````

````{py:method} _select_thompson_sampling(context_key)
:canonical: src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit._select_thompson_sampling

```{autodoc2-docstring} src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit._select_thompson_sampling
```

````

````{py:method} _select_epsilon_greedy(context_key)
:canonical: src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit._select_epsilon_greedy

```{autodoc2-docstring} src.pipeline.rl.meta.contextual_bandits.WeightContextualBandit._select_epsilon_greedy
```

````

`````
