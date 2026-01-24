# {py:mod}`src.models.policies.classical.hybrid`

```{py:module} src.models.policies.classical.hybrid
```

```{autodoc2-docstring} src.models.policies.classical.hybrid
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuralHeuristicHybrid <src.models.policies.classical.hybrid.NeuralHeuristicHybrid>`
  - ```{autodoc2-docstring} src.models.policies.classical.hybrid.NeuralHeuristicHybrid
    :summary:
    ```
````

### API

`````{py:class} NeuralHeuristicHybrid(neural_policy: logic.src.models.policies.base.ConstructivePolicy, heuristic_policy: typing.Union[logic.src.models.policies.classical.alns.ALNSPolicy, logic.src.models.policies.classical.hgs.HGSPolicy], **kwargs)
:canonical: src.models.policies.classical.hybrid.NeuralHeuristicHybrid

Bases: {py:obj}`logic.src.models.policies.base.ConstructivePolicy`

```{autodoc2-docstring} src.models.policies.classical.hybrid.NeuralHeuristicHybrid
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.classical.hybrid.NeuralHeuristicHybrid.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'greedy', **kwargs) -> dict
:canonical: src.models.policies.classical.hybrid.NeuralHeuristicHybrid.forward

```{autodoc2-docstring} src.models.policies.classical.hybrid.NeuralHeuristicHybrid.forward
```

````

`````
