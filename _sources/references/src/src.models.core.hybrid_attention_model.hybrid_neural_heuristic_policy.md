# {py:mod}`src.models.core.hybrid_attention_model.hybrid_neural_heuristic_policy`

```{py:module} src.models.core.hybrid_attention_model.hybrid_neural_heuristic_policy
```

```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_neural_heuristic_policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuralHeuristicHybrid <src.models.core.hybrid_attention_model.hybrid_neural_heuristic_policy.NeuralHeuristicHybrid>`
  - ```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_neural_heuristic_policy.NeuralHeuristicHybrid
    :summary:
    ```
````

### API

`````{py:class} NeuralHeuristicHybrid(neural_policy: logic.src.models.common.autoregressive.policy.AutoregressivePolicy, heuristic_policy: typing.Union[logic.src.models.policies.alns.VectorizedALNS, logic.src.models.policies.hgs.VectorizedHGS], **kwargs)
:canonical: src.models.core.hybrid_attention_model.hybrid_neural_heuristic_policy.NeuralHeuristicHybrid

Bases: {py:obj}`logic.src.models.common.autoregressive.policy.AutoregressivePolicy`

```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_neural_heuristic_policy.NeuralHeuristicHybrid
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_neural_heuristic_policy.NeuralHeuristicHybrid.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, strategy: str = 'greedy', num_starts: int = 1, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.core.hybrid_attention_model.hybrid_neural_heuristic_policy.NeuralHeuristicHybrid.forward

```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_neural_heuristic_policy.NeuralHeuristicHybrid.forward
```

````

`````
