# {py:mod}`src.models.common.hybrid`

```{py:module} src.models.common.hybrid
```

```{autodoc2-docstring} src.models.common.hybrid
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuralHeuristicHybrid <src.models.common.hybrid.NeuralHeuristicHybrid>`
  - ```{autodoc2-docstring} src.models.common.hybrid.NeuralHeuristicHybrid
    :summary:
    ```
````

### API

`````{py:class} NeuralHeuristicHybrid(neural_policy: logic.src.models.common.autoregressive_policy.AutoregressivePolicy, heuristic_policy: typing.Union[logic.src.models.policies.alns.VectorizedALNS, logic.src.models.policies.hgs.VectorizedHGS], **kwargs)
:canonical: src.models.common.hybrid.NeuralHeuristicHybrid

Bases: {py:obj}`logic.src.models.common.autoregressive_policy.AutoregressivePolicy`

```{autodoc2-docstring} src.models.common.hybrid.NeuralHeuristicHybrid
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.hybrid.NeuralHeuristicHybrid.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, strategy: str = 'greedy', num_starts: int = 1, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.common.hybrid.NeuralHeuristicHybrid.forward

```{autodoc2-docstring} src.models.common.hybrid.NeuralHeuristicHybrid.forward
```

````

`````
