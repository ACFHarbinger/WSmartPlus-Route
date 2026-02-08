# {py:mod}`src.models.subnets.embeddings.positional`

```{py:module} src.models.subnets.embeddings.positional
```

```{autodoc2-docstring} src.models.subnets.embeddings.positional
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.models.subnets.embeddings.positional.absolute_positional_embedding
src.models.subnets.embeddings.positional.cyclic_positional_embedding
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`pos_init_embedding <src.models.subnets.embeddings.positional.pos_init_embedding>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.positional.pos_init_embedding
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`POSITIONAL_EMBEDDING_REGISTRY <src.models.subnets.embeddings.positional.POSITIONAL_EMBEDDING_REGISTRY>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.positional.POSITIONAL_EMBEDDING_REGISTRY
    :summary:
    ```
* - {py:obj}`__all__ <src.models.subnets.embeddings.positional.__all__>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.positional.__all__
    :summary:
    ```
````

### API

````{py:function} pos_init_embedding(pos_name: str, embed_dim: int, **kwargs) -> torch.nn.Module
:canonical: src.models.subnets.embeddings.positional.pos_init_embedding

```{autodoc2-docstring} src.models.subnets.embeddings.positional.pos_init_embedding
```
````

````{py:data} POSITIONAL_EMBEDDING_REGISTRY
:canonical: src.models.subnets.embeddings.positional.POSITIONAL_EMBEDDING_REGISTRY
:value: >
   None

```{autodoc2-docstring} src.models.subnets.embeddings.positional.POSITIONAL_EMBEDDING_REGISTRY
```

````

````{py:data} __all__
:canonical: src.models.subnets.embeddings.positional.__all__
:value: >
   ['AbsolutePositionalEmbedding', 'CyclicPositionalEmbedding', 'pos_init_embedding']

```{autodoc2-docstring} src.models.subnets.embeddings.positional.__all__
```

````
