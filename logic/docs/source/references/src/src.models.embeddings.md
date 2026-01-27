# {py:mod}`src.models.embeddings`

```{py:module} src.models.embeddings
```

```{autodoc2-docstring} src.models.embeddings
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.models.embeddings.cvrpp
src.models.embeddings.vrpp
src.models.embeddings.wcvrp
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_init_embedding <src.models.embeddings.get_init_embedding>`
  - ```{autodoc2-docstring} src.models.embeddings.get_init_embedding
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`INIT_EMBEDDING_REGISTRY <src.models.embeddings.INIT_EMBEDDING_REGISTRY>`
  - ```{autodoc2-docstring} src.models.embeddings.INIT_EMBEDDING_REGISTRY
    :summary:
    ```
* - {py:obj}`__all__ <src.models.embeddings.__all__>`
  - ```{autodoc2-docstring} src.models.embeddings.__all__
    :summary:
    ```
````

### API

````{py:data} INIT_EMBEDDING_REGISTRY
:canonical: src.models.embeddings.INIT_EMBEDDING_REGISTRY
:value: >
   None

```{autodoc2-docstring} src.models.embeddings.INIT_EMBEDDING_REGISTRY
```

````

````{py:function} get_init_embedding(env_name: str, embed_dim: int = 128) -> torch.nn.Module
:canonical: src.models.embeddings.get_init_embedding

```{autodoc2-docstring} src.models.embeddings.get_init_embedding
```
````

````{py:data} __all__
:canonical: src.models.embeddings.__all__
:value: >
   ['VRPPInitEmbedding', 'CVRPPInitEmbedding', 'WCVRPInitEmbedding', 'INIT_EMBEDDING_REGISTRY', 'get_in...

```{autodoc2-docstring} src.models.embeddings.__all__
```

````
