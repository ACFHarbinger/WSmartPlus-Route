# {py:mod}`src.models.core.neuopt.decoder`

```{py:module} src.models.core.neuopt.decoder
```

```{autodoc2-docstring} src.models.core.neuopt.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuOptDecoder <src.models.core.neuopt.decoder.NeuOptDecoder>`
  - ```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder
    :summary:
    ```
````

### API

`````{py:class} NeuOptDecoder(embed_dim: int = 128, tanh_clipping: float = 6.0, seed: int = 42, **kwargs: typing.Any)
:canonical: src.models.core.neuopt.decoder.NeuOptDecoder

Bases: {py:obj}`logic.src.models.common.improvement.decoder.ImprovementDecoder`

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder.__init__
```

````{py:property} device
:canonical: src.models.core.neuopt.decoder.NeuOptDecoder.device
:type: torch.device

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder.device
```

````

````{py:method} __getstate__() -> typing.Dict[str, typing.Any]
:canonical: src.models.core.neuopt.decoder.NeuOptDecoder.__getstate__

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder.__getstate__
```

````

````{py:method} __setstate__(state: typing.Dict[str, typing.Any]) -> None
:canonical: src.models.core.neuopt.decoder.NeuOptDecoder.__setstate__

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder.__setstate__
```

````

````{py:method} _load_from_state_dict(state_dict: typing.Dict[str, typing.Any], prefix: str, local_metadata: typing.Dict[str, typing.Any], strict: bool, missing_keys: list[str], unexpected_keys: list[str], error_msgs: list[str]) -> None
:canonical: src.models.core.neuopt.decoder.NeuOptDecoder._load_from_state_dict

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder._load_from_state_dict
```

````

````{py:method} _compute_stream_scores(q_mu: torch.Tensor, q_lambda: torch.Tensor, h: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.core.neuopt.decoder.NeuOptDecoder._compute_stream_scores

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder._compute_stream_scores
```

````

````{py:method} _get_tour_successor(td: tensordict.TensorDict, x1: torch.Tensor, bs: int, n: int) -> torch.Tensor
:canonical: src.models.core.neuopt.decoder.NeuOptDecoder._get_tour_successor

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder._get_tour_successor
```

````

````{py:method} _compute_cyclic_rank_mask(td: tensordict.TensorDict, x1: torch.Tensor, bs: int, n: int, xj: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.core.neuopt.decoder.NeuOptDecoder._compute_cyclic_rank_mask

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder._compute_cyclic_rank_mask
```

````

````{py:method} forward(td: tensordict.TensorDict, embeddings: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]], env: logic.src.envs.base.base.RL4COEnvBase, **kwargs: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.core.neuopt.decoder.NeuOptDecoder.forward

```{autodoc2-docstring} src.models.core.neuopt.decoder.NeuOptDecoder.forward
```

````

`````
