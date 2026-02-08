# {py:mod}`src.models.nargnn.policy`

```{py:module} src.models.nargnn.policy
```

```{autodoc2-docstring} src.models.nargnn.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NARGNNPolicy <src.models.nargnn.policy.NARGNNPolicy>`
  - ```{autodoc2-docstring} src.models.nargnn.policy.NARGNNPolicy
    :summary:
    ```
````

### API

`````{py:class} NARGNNPolicy(encoder: typing.Optional[logic.src.models.common.nonautoregressive_policy.NonAutoregressiveEncoder] = None, decoder: typing.Optional[logic.src.models.common.nonautoregressive_policy.NonAutoregressiveDecoder] = None, embed_dim: int = 64, env_name: str = 'tsp', init_embedding: typing.Optional[torch.nn.Module] = None, edge_embedding: typing.Optional[torch.nn.Module] = None, graph_network: typing.Optional[torch.nn.Module] = None, heatmap_generator: typing.Optional[torch.nn.Module] = None, num_layers_heatmap_generator: int = 5, num_layers_graph_encoder: int = 15, act_fn: str = 'silu', agg_fn: str = 'mean', linear_bias: bool = True, train_decode_type: str = 'sampling', val_decode_type: str = 'greedy', test_decode_type: str = 'greedy', **constructive_policy_kw)
:canonical: src.models.nargnn.policy.NARGNNPolicy

Bases: {py:obj}`logic.src.models.common.nonautoregressive_policy.NonAutoregressivePolicy`

```{autodoc2-docstring} src.models.nargnn.policy.NARGNNPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.nargnn.policy.NARGNNPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, num_starts: int = 1, phase: str = 'test', **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.nargnn.policy.NARGNNPolicy.forward

```{autodoc2-docstring} src.models.nargnn.policy.NARGNNPolicy.forward
```

````

`````
