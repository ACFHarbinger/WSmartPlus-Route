# {py:mod}`src.models.core.attention_model.model`

```{py:module} src.models.core.attention_model.model
```

```{autodoc2-docstring} src.models.core.attention_model.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_ContextEmbedderAdapter <src.models.core.attention_model.model._ContextEmbedderAdapter>`
  - ```{autodoc2-docstring} src.models.core.attention_model.model._ContextEmbedderAdapter
    :summary:
    ```
* - {py:obj}`AttentionModel <src.models.core.attention_model.model.AttentionModel>`
  - ```{autodoc2-docstring} src.models.core.attention_model.model.AttentionModel
    :summary:
    ```
````

### API

`````{py:class} _ContextEmbedderAdapter(target: torch.nn.Module)
:canonical: src.models.core.attention_model.model._ContextEmbedderAdapter

```{autodoc2-docstring} src.models.core.attention_model.model._ContextEmbedderAdapter
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.attention_model.model._ContextEmbedderAdapter.__init__
```

````{py:method} __call__(*args: typing.Any, **kwargs: typing.Any) -> typing.Any
:canonical: src.models.core.attention_model.model._ContextEmbedderAdapter.__call__

```{autodoc2-docstring} src.models.core.attention_model.model._ContextEmbedderAdapter.__call__
```

````

````{py:method} forward(*args: typing.Any, **kwargs: typing.Any) -> typing.Any
:canonical: src.models.core.attention_model.model._ContextEmbedderAdapter.forward

```{autodoc2-docstring} src.models.core.attention_model.model._ContextEmbedderAdapter.forward
```

````

````{py:method} init_node_embeddings(nodes: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any
:canonical: src.models.core.attention_model.model._ContextEmbedderAdapter.init_node_embeddings

```{autodoc2-docstring} src.models.core.attention_model.model._ContextEmbedderAdapter.init_node_embeddings
```

````

````{py:method} __getattr__(name: str) -> typing.Any
:canonical: src.models.core.attention_model.model._ContextEmbedderAdapter.__getattr__

```{autodoc2-docstring} src.models.core.attention_model.model._ContextEmbedderAdapter.__getattr__
```

````

````{py:method} __deepcopy__(memo: typing.Dict[int, typing.Any]) -> typing.Any
:canonical: src.models.core.attention_model.model._ContextEmbedderAdapter.__deepcopy__

```{autodoc2-docstring} src.models.core.attention_model.model._ContextEmbedderAdapter.__deepcopy__
```

````

````{py:method} __setattr__(name: str, val: typing.Any) -> None
:canonical: src.models.core.attention_model.model._ContextEmbedderAdapter.__setattr__

````

`````

`````{py:class} AttentionModel(embed_dim: int = 128, hidden_dim: int = 512, problem: typing.Any = 'ptp', component_factory: typing.Optional[logic.src.models.subnets.factories.NeuralComponentFactory] = None, n_encode_layers: int = 3, n_encode_sublayers: typing.Optional[int] = None, n_decode_layers: typing.Optional[int] = None, dropout_rate: float = 0.1, aggregation: str = 'sum', aggregation_graph: str = 'avg', tanh_clipping: float = TANH_CLIPPING, mask_inner: bool = True, mask_logits: bool = True, mask_graph: bool = False, norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, n_heads: int = 8, checkpoint_encoder: bool = False, shrink_size: typing.Optional[int] = None, pomo_size: int = 0, temporal_horizon: int = 0, spatial_bias: bool = False, spatial_bias_scale: float = 1.0, entropy_weight: float = 0.0, predictor_layers: typing.Optional[int] = None, connection_type: str = 'residual', hyper_expansion: int = FEED_FORWARD_EXPANSION, decoder_type: str = 'attention', **kwargs: typing.Any)
:canonical: src.models.core.attention_model.model.AttentionModel

Bases: {py:obj}`logic.src.models.core.attention_model.policy.AttentionModelPolicy`, {py:obj}`logic.src.models.core.attention_model.decoding.DecodingMixin`

```{autodoc2-docstring} src.models.core.attention_model.model.AttentionModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.attention_model.model.AttentionModel.__init__
```

````{py:method} _resolve_activation_config(legacy_factory: bool, kwargs: typing.Dict[str, typing.Any]) -> logic.src.configs.models.activation_function.ActivationConfig
:canonical: src.models.core.attention_model.model.AttentionModel._resolve_activation_config
:staticmethod:

```{autodoc2-docstring} src.models.core.attention_model.model.AttentionModel._resolve_activation_config
```

````

````{py:method} _configure_legacy_embedding(temporal_horizon: int, embed_dim: int) -> None
:canonical: src.models.core.attention_model.model.AttentionModel._configure_legacy_embedding

```{autodoc2-docstring} src.models.core.attention_model.model.AttentionModel._configure_legacy_embedding
```

````

````{py:property} is_ptp
:canonical: src.models.core.attention_model.model.AttentionModel.is_ptp
:type: bool

```{autodoc2-docstring} src.models.core.attention_model.model.AttentionModel.is_ptp
```

````

````{py:property} context_embedder
:canonical: src.models.core.attention_model.model.AttentionModel.context_embedder
:type: typing.Any

```{autodoc2-docstring} src.models.core.attention_model.model.AttentionModel.context_embedder
```

````

````{py:method} _get_initial_embeddings(input: typing.Any) -> typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]]
:canonical: src.models.core.attention_model.model.AttentionModel._get_initial_embeddings

```{autodoc2-docstring} src.models.core.attention_model.model.AttentionModel._get_initial_embeddings
```

````

````{py:method} _aggregate_graph_context(outputs: torch.Tensor) -> torch.Tensor
:canonical: src.models.core.attention_model.model.AttentionModel._aggregate_graph_context

```{autodoc2-docstring} src.models.core.attention_model.model.AttentionModel._aggregate_graph_context
```

````

````{py:method} precompute_fixed(input: typing.Any, edges: typing.Optional[torch.Tensor] = None) -> typing.Any
:canonical: src.models.core.attention_model.model.AttentionModel.precompute_fixed

```{autodoc2-docstring} src.models.core.attention_model.model.AttentionModel.precompute_fixed
```

````

````{py:method} expand(t: typing.Union[torch.Tensor, logic.src.interfaces.tensor_dict_like.ITensorDictLike, None]) -> typing.Any
:canonical: src.models.core.attention_model.model.AttentionModel.expand

```{autodoc2-docstring} src.models.core.attention_model.model.AttentionModel.expand
```

````

````{py:method} forward(td: typing.Union[tensordict.TensorDict, typing.Dict[str, typing.Any]], env: typing.Optional[typing.Any] = None, strategy: typing.Optional[str] = None, num_starts: int = 1, actions: typing.Optional[torch.Tensor] = None, start_nodes: typing.Optional[torch.Tensor] = None, return_pi: bool = False, pad: bool = False, mask: typing.Optional[torch.Tensor] = None, expert_pi: typing.Optional[torch.Tensor] = None, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.models.core.attention_model.model.AttentionModel.forward

```{autodoc2-docstring} src.models.core.attention_model.model.AttentionModel.forward
```

````

````{py:method} _load_from_state_dict(state_dict: typing.Dict[str, typing.Any], prefix: str, local_metadata: typing.Dict[str, typing.Any], strict: bool, missing_keys: typing.List[str], unexpected_keys: typing.List[str], error_msgs: typing.List[str]) -> None
:canonical: src.models.core.attention_model.model.AttentionModel._load_from_state_dict

```{autodoc2-docstring} src.models.core.attention_model.model.AttentionModel._load_from_state_dict
```

````

````{py:method} load_state_dict(state_dict: typing.Dict[str, typing.Any], strict: bool = True, assign: bool = False) -> typing.Any
:canonical: src.models.core.attention_model.model.AttentionModel.load_state_dict

```{autodoc2-docstring} src.models.core.attention_model.model.AttentionModel.load_state_dict
```

````

`````
