# {py:mod}`src.utils.decoding.evaluate`

```{py:module} src.utils.decoding.evaluate
```

```{autodoc2-docstring} src.utils.decoding.evaluate
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Evaluate <src.utils.decoding.evaluate.Evaluate>`
  - ```{autodoc2-docstring} src.utils.decoding.evaluate.Evaluate
    :summary:
    ```
````

### API

`````{py:class} Evaluate(actions: typing.Optional[torch.Tensor] = None, **kwargs)
:canonical: src.utils.decoding.evaluate.Evaluate

Bases: {py:obj}`src.utils.decoding.base.DecodingStrategy`

```{autodoc2-docstring} src.utils.decoding.evaluate.Evaluate
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.decoding.evaluate.Evaluate.__init__
```

````{py:method} step(logits: torch.Tensor, mask: torch.Tensor, td: typing.Optional[tensordict.TensorDict] = None) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.utils.decoding.evaluate.Evaluate.step

```{autodoc2-docstring} src.utils.decoding.evaluate.Evaluate.step
```

````

````{py:method} reset()
:canonical: src.utils.decoding.evaluate.Evaluate.reset

```{autodoc2-docstring} src.utils.decoding.evaluate.Evaluate.reset
```

````

`````
