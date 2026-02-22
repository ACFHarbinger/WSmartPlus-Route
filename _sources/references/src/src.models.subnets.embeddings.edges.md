# {py:mod}`src.models.subnets.embeddings.edges`

```{py:module} src.models.subnets.embeddings.edges
```

```{autodoc2-docstring} src.models.subnets.embeddings.edges
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.models.subnets.embeddings.edges.cvrpp
src.models.subnets.embeddings.edges.none
src.models.subnets.embeddings.edges.wcvrp
src.models.subnets.embeddings.edges.tsp
src.models.subnets.embeddings.edges.base
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_edge_embedding <src.models.subnets.embeddings.edges.get_edge_embedding>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.edges.get_edge_embedding
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EDGE_EMBEDDING_REGISTRY <src.models.subnets.embeddings.edges.EDGE_EMBEDDING_REGISTRY>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.edges.EDGE_EMBEDDING_REGISTRY
    :summary:
    ```
* - {py:obj}`__all__ <src.models.subnets.embeddings.edges.__all__>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.edges.__all__
    :summary:
    ```
````

### API

````{py:data} EDGE_EMBEDDING_REGISTRY
:canonical: src.models.subnets.embeddings.edges.EDGE_EMBEDDING_REGISTRY
:value: >
   None

```{autodoc2-docstring} src.models.subnets.embeddings.edges.EDGE_EMBEDDING_REGISTRY
```

````

````{py:function} get_edge_embedding(env_name: str, embed_dim: int = 128, **kwargs) -> torch.nn.Module
:canonical: src.models.subnets.embeddings.edges.get_edge_embedding

```{autodoc2-docstring} src.models.subnets.embeddings.edges.get_edge_embedding
```
````

````{py:data} __all__
:canonical: src.models.subnets.embeddings.edges.__all__
:value: >
   ['EdgeEmbedding', 'CVRPPEdgeEmbedding', 'NoEdgeEmbedding', 'TSPEdgeEmbedding', 'WCVRPEdgeEmbedding',...

```{autodoc2-docstring} src.models.subnets.embeddings.edges.__all__
```

````
