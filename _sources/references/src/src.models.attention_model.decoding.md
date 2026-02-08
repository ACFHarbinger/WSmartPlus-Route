# {py:mod}`src.models.attention_model.decoding`

```{py:module} src.models.attention_model.decoding
```

```{autodoc2-docstring} src.models.attention_model.decoding
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DecodingMixin <src.models.attention_model.decoding.DecodingMixin>`
  - ```{autodoc2-docstring} src.models.attention_model.decoding.DecodingMixin
    :summary:
    ```
````

### API

`````{py:class} DecodingMixin()
:canonical: src.models.attention_model.decoding.DecodingMixin

```{autodoc2-docstring} src.models.attention_model.decoding.DecodingMixin
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.attention_model.decoding.DecodingMixin.__init__
```

````{py:method} set_decode_type(decode_type: str, temp: typing.Optional[float] = None)
:canonical: src.models.attention_model.decoding.DecodingMixin.set_decode_type

```{autodoc2-docstring} src.models.attention_model.decoding.DecodingMixin.set_decode_type
```

````

````{py:method} beam_search(*args: typing.Any, **kwargs: typing.Any)
:canonical: src.models.attention_model.decoding.DecodingMixin.beam_search

```{autodoc2-docstring} src.models.attention_model.decoding.DecodingMixin.beam_search
```

````

````{py:method} propose_expansions(beam: typing.Any, fixed: typing.Any, expand_size: typing.Optional[int] = None, normalize: bool = False, max_calc_batch_size: int = 4096)
:canonical: src.models.attention_model.decoding.DecodingMixin.propose_expansions

```{autodoc2-docstring} src.models.attention_model.decoding.DecodingMixin.propose_expansions
```

````

````{py:method} sample_many(input: typing.Dict[str, typing.Any], cost_weights: typing.Optional[torch.Tensor] = None, batch_rep: int = 1, iter_rep: int = 1) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.attention_model.decoding.DecodingMixin.sample_many

```{autodoc2-docstring} src.models.attention_model.decoding.DecodingMixin.sample_many
```

````

`````
