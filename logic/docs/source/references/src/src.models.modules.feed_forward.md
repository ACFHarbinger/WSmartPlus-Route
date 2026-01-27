# {py:mod}`src.models.modules.feed_forward`

```{py:module} src.models.modules.feed_forward
```

```{autodoc2-docstring} src.models.modules.feed_forward
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FeedForward <src.models.modules.feed_forward.FeedForward>`
  - ```{autodoc2-docstring} src.models.modules.feed_forward.FeedForward
    :summary:
    ```
````

### API

`````{py:class} FeedForward(input_dim: int, output_dim: int, bias: bool = True)
:canonical: src.models.modules.feed_forward.FeedForward

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.modules.feed_forward.FeedForward
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.modules.feed_forward.FeedForward.__init__
```

````{py:method} init_parameters() -> None
:canonical: src.models.modules.feed_forward.FeedForward.init_parameters

```{autodoc2-docstring} src.models.modules.feed_forward.FeedForward.init_parameters
```

````

````{py:method} forward(input: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.modules.feed_forward.FeedForward.forward

```{autodoc2-docstring} src.models.modules.feed_forward.FeedForward.forward
```

````

`````
