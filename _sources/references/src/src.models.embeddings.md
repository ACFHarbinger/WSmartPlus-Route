# {py:mod}`src.models.embeddings`

```{py:module} src.models.embeddings
```

```{autodoc2-docstring} src.models.embeddings
:allowtitles:
```

## Package Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VRPPInitEmbedding <src.models.embeddings.VRPPInitEmbedding>`
  - ```{autodoc2-docstring} src.models.embeddings.VRPPInitEmbedding
    :summary:
    ```
* - {py:obj}`CVRPPInitEmbedding <src.models.embeddings.CVRPPInitEmbedding>`
  - ```{autodoc2-docstring} src.models.embeddings.CVRPPInitEmbedding
    :summary:
    ```
* - {py:obj}`WCVRPInitEmbedding <src.models.embeddings.WCVRPInitEmbedding>`
  - ```{autodoc2-docstring} src.models.embeddings.WCVRPInitEmbedding
    :summary:
    ```
````

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

`````{py:class} VRPPInitEmbedding(embed_dim: int = 128)
:canonical: src.models.embeddings.VRPPInitEmbedding

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.embeddings.VRPPInitEmbedding
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.embeddings.VRPPInitEmbedding.__init__
```

````{py:method} forward(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.models.embeddings.VRPPInitEmbedding.forward

```{autodoc2-docstring} src.models.embeddings.VRPPInitEmbedding.forward
```

````

`````

`````{py:class} CVRPPInitEmbedding(embed_dim: int = 128)
:canonical: src.models.embeddings.CVRPPInitEmbedding

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.embeddings.CVRPPInitEmbedding
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.embeddings.CVRPPInitEmbedding.__init__
```

````{py:method} forward(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.models.embeddings.CVRPPInitEmbedding.forward

```{autodoc2-docstring} src.models.embeddings.CVRPPInitEmbedding.forward
```

````

`````

`````{py:class} WCVRPInitEmbedding(embed_dim: int = 128)
:canonical: src.models.embeddings.WCVRPInitEmbedding

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.embeddings.WCVRPInitEmbedding
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.embeddings.WCVRPInitEmbedding.__init__
```

````{py:method} forward(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.models.embeddings.WCVRPInitEmbedding.forward

```{autodoc2-docstring} src.models.embeddings.WCVRPInitEmbedding.forward
```

````

`````

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
