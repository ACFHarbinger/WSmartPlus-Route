# {py:mod}`src.models.subnets.decoders.mdam.cache`

```{py:module} src.models.subnets.decoders.mdam.cache
```

```{autodoc2-docstring} src.models.subnets.decoders.mdam.cache
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PrecomputedCache <src.models.subnets.decoders.mdam.cache.PrecomputedCache>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.mdam.cache.PrecomputedCache
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_decode_probs <src.models.subnets.decoders.mdam.cache._decode_probs>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.mdam.cache._decode_probs
    :summary:
    ```
````

### API

````{py:function} _decode_probs(probs: torch.Tensor, mask: typing.Optional[torch.Tensor], decode_type: str = 'sampling') -> torch.Tensor
:canonical: src.models.subnets.decoders.mdam.cache._decode_probs

```{autodoc2-docstring} src.models.subnets.decoders.mdam.cache._decode_probs
```
````

`````{py:class} PrecomputedCache
:canonical: src.models.subnets.decoders.mdam.cache.PrecomputedCache

```{autodoc2-docstring} src.models.subnets.decoders.mdam.cache.PrecomputedCache
```

````{py:attribute} node_embeddings
:canonical: src.models.subnets.decoders.mdam.cache.PrecomputedCache.node_embeddings
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.subnets.decoders.mdam.cache.PrecomputedCache.node_embeddings
```

````

````{py:attribute} graph_context
:canonical: src.models.subnets.decoders.mdam.cache.PrecomputedCache.graph_context
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.subnets.decoders.mdam.cache.PrecomputedCache.graph_context
```

````

````{py:attribute} glimpse_key
:canonical: src.models.subnets.decoders.mdam.cache.PrecomputedCache.glimpse_key
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.subnets.decoders.mdam.cache.PrecomputedCache.glimpse_key
```

````

````{py:attribute} glimpse_val
:canonical: src.models.subnets.decoders.mdam.cache.PrecomputedCache.glimpse_val
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.subnets.decoders.mdam.cache.PrecomputedCache.glimpse_val
```

````

````{py:attribute} logit_key
:canonical: src.models.subnets.decoders.mdam.cache.PrecomputedCache.logit_key
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.subnets.decoders.mdam.cache.PrecomputedCache.logit_key
```

````

`````
