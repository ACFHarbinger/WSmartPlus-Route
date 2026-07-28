# {py:mod}`src.models.subnets.decoders.mdam.decoder`

```{py:module} src.models.subnets.decoders.mdam.decoder
```

```{autodoc2-docstring} src.models.subnets.decoders.mdam.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MDAMDecoder <src.models.subnets.decoders.mdam.decoder.MDAMDecoder>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.mdam.decoder.MDAMDecoder
    :summary:
    ```
````

### API

`````{py:class} MDAMDecoder(embed_dim: int = 128, num_heads: int = 8, num_paths: int = 5, env_name: str = 'vrpp', mask_inner: bool = True, mask_logits: bool = True, eg_step_gap: int = 200, tanh_clipping: float = 10.0, train_strategy: str = 'sampling', val_strategy: str = 'greedy', test_strategy: str = 'greedy')
:canonical: src.models.subnets.decoders.mdam.decoder.MDAMDecoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.decoders.mdam.decoder.MDAMDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.decoders.mdam.decoder.MDAMDecoder.__init__
```

````{py:attribute} paths
:canonical: src.models.subnets.decoders.mdam.decoder.MDAMDecoder.paths
:type: torch.nn.ModuleList
:value: >
   None

```{autodoc2-docstring} src.models.subnets.decoders.mdam.decoder.MDAMDecoder.paths
```

````

````{py:attribute} W_placeholder
:canonical: src.models.subnets.decoders.mdam.decoder.MDAMDecoder.W_placeholder
:type: torch.nn.Parameter
:value: >
   None

```{autodoc2-docstring} src.models.subnets.decoders.mdam.decoder.MDAMDecoder.W_placeholder
```

````

````{py:method} forward(td: tensordict.TensorDict, embeddings: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]], env: logic.src.envs.base.base.RL4COEnvBase, strategy: typing.Optional[str] = 'greedy', **kwargs: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.decoders.mdam.decoder.MDAMDecoder.forward

```{autodoc2-docstring} src.models.subnets.decoders.mdam.decoder.MDAMDecoder.forward
```

````

````{py:method} _compute_initial_kl_divergence(td: tensordict.TensorDict, h: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.decoders.mdam.decoder.MDAMDecoder._compute_initial_kl_divergence

```{autodoc2-docstring} src.models.subnets.decoders.mdam.decoder.MDAMDecoder._compute_initial_kl_divergence
```

````

````{py:method} _decode_path(td: tensordict.TensorDict, h: torch.Tensor, env: logic.src.envs.base.base.RL4COEnvBase, attn: torch.Tensor, V: torch.Tensor, h_old: torch.Tensor, encoder: typing.Any, path_idx: int, strategy: str) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.decoders.mdam.decoder.MDAMDecoder._decode_path

```{autodoc2-docstring} src.models.subnets.decoders.mdam.decoder.MDAMDecoder._decode_path
```

````

````{py:method} _get_log_likelihood(log_probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.decoders.mdam.decoder.MDAMDecoder._get_log_likelihood

```{autodoc2-docstring} src.models.subnets.decoders.mdam.decoder.MDAMDecoder._get_log_likelihood
```

````

`````
