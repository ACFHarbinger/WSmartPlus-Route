# {py:mod}`src.models.subnets.decoders.common.cache`

```{py:module} src.models.subnets.decoders.common.cache
```

```{autodoc2-docstring} src.models.subnets.decoders.common.cache
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AttentionDecoderCache <src.models.subnets.decoders.common.cache.AttentionDecoderCache>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.common.cache.AttentionDecoderCache
    :summary:
    ```
````

### API

`````{py:class} AttentionDecoderCache
:canonical: src.models.subnets.decoders.common.cache.AttentionDecoderCache

```{autodoc2-docstring} src.models.subnets.decoders.common.cache.AttentionDecoderCache
```

````{py:attribute} node_embeddings
:canonical: src.models.subnets.decoders.common.cache.AttentionDecoderCache.node_embeddings
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.subnets.decoders.common.cache.AttentionDecoderCache.node_embeddings
```

````

````{py:attribute} graph_context
:canonical: src.models.subnets.decoders.common.cache.AttentionDecoderCache.graph_context
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.subnets.decoders.common.cache.AttentionDecoderCache.graph_context
```

````

````{py:attribute} glimpse_key
:canonical: src.models.subnets.decoders.common.cache.AttentionDecoderCache.glimpse_key
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} src.models.subnets.decoders.common.cache.AttentionDecoderCache.glimpse_key
```

````

````{py:attribute} glimpse_val
:canonical: src.models.subnets.decoders.common.cache.AttentionDecoderCache.glimpse_val
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} src.models.subnets.decoders.common.cache.AttentionDecoderCache.glimpse_val
```

````

````{py:attribute} logit_key
:canonical: src.models.subnets.decoders.common.cache.AttentionDecoderCache.logit_key
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} src.models.subnets.decoders.common.cache.AttentionDecoderCache.logit_key
```

````

````{py:method} __getitem__(key: typing.Union[int, slice, torch.Tensor]) -> src.models.subnets.decoders.common.cache.AttentionDecoderCache
:canonical: src.models.subnets.decoders.common.cache.AttentionDecoderCache.__getitem__

```{autodoc2-docstring} src.models.subnets.decoders.common.cache.AttentionDecoderCache.__getitem__
```

````

````{py:property} context_node_projected
:canonical: src.models.subnets.decoders.common.cache.AttentionDecoderCache.context_node_projected
:type: torch.Tensor

```{autodoc2-docstring} src.models.subnets.decoders.common.cache.AttentionDecoderCache.context_node_projected
```

````

`````
