# {py:mod}`src.models.subnets.decoders.mdam.path`

```{py:module} src.models.subnets.decoders.mdam.path
```

```{autodoc2-docstring} src.models.subnets.decoders.mdam.path
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MDAMPath <src.models.subnets.decoders.mdam.path.MDAMPath>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.mdam.path.MDAMPath
    :summary:
    ```
````

### API

`````{py:class} MDAMPath(embed_dim: int, env_name: str, num_heads: int, tanh_clipping: float = 10.0, mask_inner: bool = True, mask_logits: bool = True)
:canonical: src.models.subnets.decoders.mdam.path.MDAMPath

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.decoders.mdam.path.MDAMPath
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.decoders.mdam.path.MDAMPath.__init__
```

````{py:method} precompute(h_embed: torch.Tensor, num_steps: int = 1) -> src.models.subnets.decoders.mdam.cache.PrecomputedCache
:canonical: src.models.subnets.decoders.mdam.path.MDAMPath.precompute

```{autodoc2-docstring} src.models.subnets.decoders.mdam.path.MDAMPath.precompute
```

````

````{py:method} _make_heads(v: torch.Tensor, num_steps: typing.Optional[int] = None) -> torch.Tensor
:canonical: src.models.subnets.decoders.mdam.path.MDAMPath._make_heads

```{autodoc2-docstring} src.models.subnets.decoders.mdam.path.MDAMPath._make_heads
```

````

````{py:method} get_logprobs(fixed: src.models.subnets.decoders.mdam.cache.PrecomputedCache, td: tensordict.TensorDict, dynamic_embed: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor], path_index: int) -> typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]]
:canonical: src.models.subnets.decoders.mdam.path.MDAMPath.get_logprobs

```{autodoc2-docstring} src.models.subnets.decoders.mdam.path.MDAMPath.get_logprobs
```

````

`````
