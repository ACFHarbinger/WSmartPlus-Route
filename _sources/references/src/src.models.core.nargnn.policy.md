# {py:mod}`src.models.core.nargnn.policy`

```{py:module} src.models.core.nargnn.policy
```

```{autodoc2-docstring} src.models.core.nargnn.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NARGNNPolicy <src.models.core.nargnn.policy.NARGNNPolicy>`
  - ```{autodoc2-docstring} src.models.core.nargnn.policy.NARGNNPolicy
    :summary:
    ```
````

### API

`````{py:class} NARGNNPolicy(encoder: typing.Optional[logic.src.models.common.non_autoregressive.encoder.NonAutoregressiveEncoder] = None, decoder: typing.Optional[logic.src.models.common.non_autoregressive.decoder.NonAutoregressiveDecoder] = None, embed_dim: int = 64, env_name: str = 'tsp', init_embedding: typing.Optional[torch.nn.Module] = None, edge_embedding: typing.Optional[torch.nn.Module] = None, graph_network: typing.Optional[torch.nn.Module] = None, heatmap_generator: typing.Optional[torch.nn.Module] = None, num_layers_heatmap_generator: int = 5, num_layers_graph_encoder: int = 15, act_fn: str = 'silu', agg_fn: str = 'mean', linear_bias: bool = True, train_strategy: str = 'sampling', val_strategy: str = 'greedy', test_strategy: str = 'greedy', **constructive_policy_kw: typing.Any)
:canonical: src.models.core.nargnn.policy.NARGNNPolicy

Bases: {py:obj}`logic.src.models.common.non_autoregressive.policy.NonAutoregressivePolicy`

```{autodoc2-docstring} src.models.core.nargnn.policy.NARGNNPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.nargnn.policy.NARGNNPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.base.RL4COEnvBase] = None, num_starts: int = 1, phase: str = 'test', **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.models.core.nargnn.policy.NARGNNPolicy.forward

```{autodoc2-docstring} src.models.core.nargnn.policy.NARGNNPolicy.forward
```

````

`````
