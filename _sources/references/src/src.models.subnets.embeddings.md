# {py:mod}`src.models.subnets.embeddings`

```{py:module} src.models.subnets.embeddings
```

```{autodoc2-docstring} src.models.subnets.embeddings
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

src.models.subnets.embeddings.context
src.models.subnets.embeddings.state
src.models.subnets.embeddings.edges
src.models.subnets.embeddings.positional
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.models.subnets.embeddings.matnet
src.models.subnets.embeddings.static
src.models.subnets.embeddings.dynamic
src.models.subnets.embeddings.wcvrp
src.models.subnets.embeddings.vrpp
src.models.subnets.embeddings.cvrpp
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_init_embedding <src.models.subnets.embeddings.get_init_embedding>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.get_init_embedding
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`INIT_EMBEDDING_REGISTRY <src.models.subnets.embeddings.INIT_EMBEDDING_REGISTRY>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.INIT_EMBEDDING_REGISTRY
    :summary:
    ```
* - {py:obj}`DYNAMIC_EMBEDDING_REGISTRY <src.models.subnets.embeddings.DYNAMIC_EMBEDDING_REGISTRY>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.DYNAMIC_EMBEDDING_REGISTRY
    :summary:
    ```
* - {py:obj}`__all__ <src.models.subnets.embeddings.__all__>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.__all__
    :summary:
    ```
````

### API

````{py:data} INIT_EMBEDDING_REGISTRY
:canonical: src.models.subnets.embeddings.INIT_EMBEDDING_REGISTRY
:value: >
   None

```{autodoc2-docstring} src.models.subnets.embeddings.INIT_EMBEDDING_REGISTRY
```

````

````{py:data} DYNAMIC_EMBEDDING_REGISTRY
:canonical: src.models.subnets.embeddings.DYNAMIC_EMBEDDING_REGISTRY
:value: >
   None

```{autodoc2-docstring} src.models.subnets.embeddings.DYNAMIC_EMBEDDING_REGISTRY
```

````

````{py:function} get_init_embedding(env_name: str, embed_dim: int = 128) -> torch.nn.Module
:canonical: src.models.subnets.embeddings.get_init_embedding

```{autodoc2-docstring} src.models.subnets.embeddings.get_init_embedding
```
````

````{py:data} __all__
:canonical: src.models.subnets.embeddings.__all__
:value: >
   ['VRPPInitEmbedding', 'CVRPPInitEmbedding', 'WCVRPInitEmbedding', 'EnvState', 'VRPPState', 'CVRPPSta...

```{autodoc2-docstring} src.models.subnets.embeddings.__all__
```

````
