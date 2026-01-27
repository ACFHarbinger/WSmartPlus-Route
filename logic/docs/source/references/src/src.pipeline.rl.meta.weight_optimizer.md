# {py:mod}`src.pipeline.rl.meta.weight_optimizer`

```{py:module} src.pipeline.rl.meta.weight_optimizer
```

```{autodoc2-docstring} src.pipeline.rl.meta.weight_optimizer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RewardWeightOptimizer <src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer>`
  - ```{autodoc2-docstring} src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer
    :summary:
    ```
````

### API

`````{py:class} RewardWeightOptimizer(model_class: torch.nn.Module, initial_weights: typing.Dict[str, float], history_length: int = 10, hidden_size: int = 64, lr: float = 0.001, device: str = 'cpu', meta_batch_size: int = 8, min_weights: typing.Optional[typing.List[float]] = None, max_weights: typing.Optional[typing.List[float]] = None, meta_optimizer: str = 'adam', **kwargs)
:canonical: src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer

Bases: {py:obj}`logic.src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy`

```{autodoc2-docstring} src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.__init__
```

````{py:method} propose_weights(context=None)
:canonical: src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.propose_weights

```{autodoc2-docstring} src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.propose_weights
```

````

````{py:method} feedback(reward, metrics, day=None, step=None)
:canonical: src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.feedback

```{autodoc2-docstring} src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.feedback
```

````

````{py:method} update_histories(performance_metrics, reward)
:canonical: src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.update_histories

```{autodoc2-docstring} src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.update_histories
```

````

````{py:method} prepare_meta_learning_batch()
:canonical: src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.prepare_meta_learning_batch

```{autodoc2-docstring} src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.prepare_meta_learning_batch
```

````

````{py:method} meta_learning_step()
:canonical: src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.meta_learning_step

```{autodoc2-docstring} src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.meta_learning_step
```

````

````{py:method} recommend_weights()
:canonical: src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.recommend_weights

```{autodoc2-docstring} src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.recommend_weights
```

````

````{py:method} update_weights_internal(force_update=False)
:canonical: src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.update_weights_internal

```{autodoc2-docstring} src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.update_weights_internal
```

````

````{py:method} get_current_weights()
:canonical: src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.get_current_weights

```{autodoc2-docstring} src.pipeline.rl.meta.weight_optimizer.RewardWeightOptimizer.get_current_weights
```

````

`````
