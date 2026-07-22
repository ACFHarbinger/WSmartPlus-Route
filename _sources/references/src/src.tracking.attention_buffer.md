# {py:mod}`src.tracking.attention_buffer`

```{py:module} src.tracking.attention_buffer
```

```{autodoc2-docstring} src.tracking.attention_buffer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AttentionRingBuffer <src.tracking.attention_buffer.AttentionRingBuffer>`
  - ```{autodoc2-docstring} src.tracking.attention_buffer.AttentionRingBuffer
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`install_attention_ring_buffer <src.tracking.attention_buffer.install_attention_ring_buffer>`
  - ```{autodoc2-docstring} src.tracking.attention_buffer.install_attention_ring_buffer
    :summary:
    ```
* - {py:obj}`ensure_attention_buffer <src.tracking.attention_buffer.ensure_attention_buffer>`
  - ```{autodoc2-docstring} src.tracking.attention_buffer.ensure_attention_buffer
    :summary:
    ```
````

### API

`````{py:class} AttentionRingBuffer(max_history: int = 64, max_matrix_dim: int = 128)
:canonical: src.tracking.attention_buffer.AttentionRingBuffer

```{autodoc2-docstring} src.tracking.attention_buffer.AttentionRingBuffer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.attention_buffer.AttentionRingBuffer.__init__
```

````{py:property} decode_step
:canonical: src.tracking.attention_buffer.AttentionRingBuffer.decode_step
:type: int

```{autodoc2-docstring} src.tracking.attention_buffer.AttentionRingBuffer.decode_step
```

````

````{py:method} bump_decode_step() -> None
:canonical: src.tracking.attention_buffer.AttentionRingBuffer.bump_decode_step

```{autodoc2-docstring} src.tracking.attention_buffer.AttentionRingBuffer.bump_decode_step
```

````

````{py:method} record(layer_idx: int, tensor: torch.Tensor, head_idx: int = 0) -> None
:canonical: src.tracking.attention_buffer.AttentionRingBuffer.record

```{autodoc2-docstring} src.tracking.attention_buffer.AttentionRingBuffer.record
```

````

````{py:method} get_snapshots() -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.tracking.attention_buffer.AttentionRingBuffer.get_snapshots

```{autodoc2-docstring} src.tracking.attention_buffer.AttentionRingBuffer.get_snapshots
```

````

````{py:method} as_layer_matrices(head_idx: int = 0) -> typing.List[typing.Tuple[int, numpy.ndarray]]
:canonical: src.tracking.attention_buffer.AttentionRingBuffer.as_layer_matrices

```{autodoc2-docstring} src.tracking.attention_buffer.AttentionRingBuffer.as_layer_matrices
```

````

````{py:method} reset(*, reset_decode_step: bool = True) -> None
:canonical: src.tracking.attention_buffer.AttentionRingBuffer.reset

```{autodoc2-docstring} src.tracking.attention_buffer.AttentionRingBuffer.reset
```

````

````{py:method} __len__() -> int
:canonical: src.tracking.attention_buffer.AttentionRingBuffer.__len__

```{autodoc2-docstring} src.tracking.attention_buffer.AttentionRingBuffer.__len__
```

````

`````

````{py:function} install_attention_ring_buffer(encoder: torch.nn.Module, buffer: src.tracking.attention_buffer.AttentionRingBuffer, head_idx: int = 0) -> typing.List[typing.Any]
:canonical: src.tracking.attention_buffer.install_attention_ring_buffer

```{autodoc2-docstring} src.tracking.attention_buffer.install_attention_ring_buffer
```
````

````{py:function} ensure_attention_buffer(model: typing.Any, head_idx: int = 0) -> typing.Optional[src.tracking.attention_buffer.AttentionRingBuffer]
:canonical: src.tracking.attention_buffer.ensure_attention_buffer

```{autodoc2-docstring} src.tracking.attention_buffer.ensure_attention_buffer
```
````
