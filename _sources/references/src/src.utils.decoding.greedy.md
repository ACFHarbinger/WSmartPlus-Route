# {py:mod}`src.utils.decoding.greedy`

```{py:module} src.utils.decoding.greedy
```

```{autodoc2-docstring} src.utils.decoding.greedy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Greedy <src.utils.decoding.greedy.Greedy>`
  - ```{autodoc2-docstring} src.utils.decoding.greedy.Greedy
    :summary:
    ```
````

### API

`````{py:class} Greedy(**kwargs)
:canonical: src.utils.decoding.greedy.Greedy

Bases: {py:obj}`src.utils.decoding.base.DecodingStrategy`

```{autodoc2-docstring} src.utils.decoding.greedy.Greedy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.decoding.greedy.Greedy.__init__
```

````{py:method} step(logits: torch.Tensor, mask: torch.Tensor, td: typing.Optional[tensordict.TensorDict] = None) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.utils.decoding.greedy.Greedy.step

```{autodoc2-docstring} src.utils.decoding.greedy.Greedy.step
```

````

`````
