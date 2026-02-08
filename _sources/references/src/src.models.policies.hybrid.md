# {py:mod}`src.models.policies.hybrid`

```{py:module} src.models.policies.hybrid
```

```{autodoc2-docstring} src.models.policies.hybrid
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuralHeuristicHybrid <src.models.policies.hybrid.NeuralHeuristicHybrid>`
  - ```{autodoc2-docstring} src.models.policies.hybrid.NeuralHeuristicHybrid
    :summary:
    ```
````

### API

`````{py:class} NeuralHeuristicHybrid(neural_policy: logic.src.models.common.autoregressive_policy.AutoregressivePolicy, heuristic_policy: typing.Union[logic.src.models.policies.alns.VectorizedALNS, logic.src.models.policies.hgs.VectorizedHGS], **kwargs)
:canonical: src.models.policies.hybrid.NeuralHeuristicHybrid

Bases: {py:obj}`logic.src.models.common.autoregressive_policy.AutoregressivePolicy`

```{autodoc2-docstring} src.models.policies.hybrid.NeuralHeuristicHybrid
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.hybrid.NeuralHeuristicHybrid.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'greedy', num_starts: int = 1, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.policies.hybrid.NeuralHeuristicHybrid.forward

```{autodoc2-docstring} src.models.policies.hybrid.NeuralHeuristicHybrid.forward
```

````

`````
