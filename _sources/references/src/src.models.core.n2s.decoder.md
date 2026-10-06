# {py:mod}`src.models.core.n2s.decoder`

```{py:module} src.models.core.n2s.decoder
```

```{autodoc2-docstring} src.models.core.n2s.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`N2SDecoder <src.models.core.n2s.decoder.N2SDecoder>`
  - ```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder
    :summary:
    ```
````

### API

`````{py:class} N2SDecoder(embed_dim: int = 128, num_heads: int = 4, tanh_clipping: float = 6.0, seed: int = 42, **kwargs: typing.Any)
:canonical: src.models.core.n2s.decoder.N2SDecoder

Bases: {py:obj}`logic.src.models.common.improvement.decoder.ImprovementDecoder`

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder.__init__
```

````{py:property} device
:canonical: src.models.core.n2s.decoder.N2SDecoder.device
:type: torch.device

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder.device
```

````

````{py:method} __getstate__() -> typing.Dict[str, typing.Any]
:canonical: src.models.core.n2s.decoder.N2SDecoder.__getstate__

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder.__getstate__
```

````

````{py:method} __setstate__(state: typing.Dict[str, typing.Any]) -> None
:canonical: src.models.core.n2s.decoder.N2SDecoder.__setstate__

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder.__setstate__
```

````

````{py:method} _load_from_state_dict(state_dict: typing.Dict[str, typing.Any], prefix: str, local_metadata: typing.Dict[str, typing.Any], strict: bool, missing_keys: list[str], unexpected_keys: list[str], error_msgs: list[str]) -> None
:canonical: src.models.core.n2s.decoder.N2SDecoder._load_from_state_dict

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder._load_from_state_dict
```

````

````{py:method} _get_tour_neighbors(td: tensordict.TensorDict, bs: int, n: int, device: torch.device) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.core.n2s.decoder.N2SDecoder._get_tour_neighbors

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder._get_tour_neighbors
```

````

````{py:method} _validate_partner_ids(p: torch.Tensor, bs: int, n: int, device: torch.device) -> torch.Tensor
:canonical: src.models.core.n2s.decoder.N2SDecoder._validate_partner_ids
:staticmethod:

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder._validate_partner_ids
```

````

````{py:method} _resolve_partner_ids(td: tensordict.TensorDict, bs: int, n: int, device: torch.device, kwargs: typing.Dict[str, typing.Any]) -> torch.Tensor
:canonical: src.models.core.n2s.decoder.N2SDecoder._resolve_partner_ids

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder._resolve_partner_ids
```

````

````{py:method} _identify_delivery_nodes(sol_tensor: typing.Optional[torch.Tensor], partner_ids: torch.Tensor, bs: int, n: int, device: torch.device, td: typing.Optional[tensordict.TensorDict] = None, validate_solution: bool = False) -> typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]]
:canonical: src.models.core.n2s.decoder.N2SDecoder._identify_delivery_nodes

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder._identify_delivery_nodes
```

````

````{py:method} forward(td: tensordict.TensorDict, embeddings: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]], env: logic.src.envs.base.base.RL4COEnvBase, **kwargs: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.core.n2s.decoder.N2SDecoder.forward

```{autodoc2-docstring} src.models.core.n2s.decoder.N2SDecoder.forward
```

````

`````
