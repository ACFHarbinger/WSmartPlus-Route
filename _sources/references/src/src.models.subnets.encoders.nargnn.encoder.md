# {py:mod}`src.models.subnets.encoders.nargnn.encoder`

```{py:module} src.models.subnets.encoders.nargnn.encoder
```

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NARGNNEncoder <src.models.subnets.encoders.nargnn.encoder.NARGNNEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.nargnn.encoder.NARGNNEncoder
    :summary:
    ```
````

### API

`````{py:class} NARGNNEncoder(embed_dim: int = 64, env_name: str = 'tsp', init_embedding: typing.Optional[torch.nn.Module] = None, edge_embedding: typing.Optional[torch.nn.Module] = None, graph_network: typing.Optional[torch.nn.Module] = None, heatmap_generator: typing.Optional[torch.nn.Module] = None, num_layers_heatmap_generator: int = 5, num_layers_graph_encoder: int = 15, act_fn: str = 'silu', agg_fn: str = 'mean', linear_bias: bool = True, k_sparse: typing.Optional[int] = None, **kwargs)
:canonical: src.models.subnets.encoders.nargnn.encoder.NARGNNEncoder

Bases: {py:obj}`logic.src.models.common.nonautoregressive_encoder.NonAutoregressiveEncoder`

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.encoder.NARGNNEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.encoder.NARGNNEncoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.encoders.nargnn.encoder.NARGNNEncoder.forward

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.encoder.NARGNNEncoder.forward
```

````

`````
