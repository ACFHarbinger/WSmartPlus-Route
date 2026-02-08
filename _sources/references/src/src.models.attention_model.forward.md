# {py:mod}`src.models.attention_model.forward`

```{py:module} src.models.attention_model.forward
```

```{autodoc2-docstring} src.models.attention_model.forward
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ForwardMixin <src.models.attention_model.forward.ForwardMixin>`
  - ```{autodoc2-docstring} src.models.attention_model.forward.ForwardMixin
    :summary:
    ```
````

### API

`````{py:class} ForwardMixin()
:canonical: src.models.attention_model.forward.ForwardMixin

```{autodoc2-docstring} src.models.attention_model.forward.ForwardMixin
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.attention_model.forward.ForwardMixin.__init__
```

````{py:method} _get_initial_embeddings(input: typing.Dict[str, torch.Tensor])
:canonical: src.models.attention_model.forward.ForwardMixin._get_initial_embeddings

```{autodoc2-docstring} src.models.attention_model.forward.ForwardMixin._get_initial_embeddings
```

````

````{py:method} forward(input: typing.Dict[str, torch.Tensor], env: typing.Optional[typing.Any] = None, decode_type: typing.Optional[str] = None, return_pi: bool = False, pad: bool = False, mask: typing.Optional[torch.Tensor] = None, expert_pi: typing.Optional[torch.Tensor] = None, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.models.attention_model.forward.ForwardMixin.forward

```{autodoc2-docstring} src.models.attention_model.forward.ForwardMixin.forward
```

````

````{py:method} precompute_fixed(input: typing.Dict[str, torch.Tensor], edges: typing.Optional[torch.Tensor])
:canonical: src.models.attention_model.forward.ForwardMixin.precompute_fixed

```{autodoc2-docstring} src.models.attention_model.forward.ForwardMixin.precompute_fixed
```

````

````{py:method} expand(t)
:canonical: src.models.attention_model.forward.ForwardMixin.expand

```{autodoc2-docstring} src.models.attention_model.forward.ForwardMixin.expand
```

````

`````
