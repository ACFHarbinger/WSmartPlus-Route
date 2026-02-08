# {py:mod}`src.models.subnets.decoders.deepaco.decoder`

```{py:module} src.models.subnets.decoders.deepaco.decoder
```

```{autodoc2-docstring} src.models.subnets.decoders.deepaco.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ACODecoder <src.models.subnets.decoders.deepaco.decoder.ACODecoder>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.deepaco.decoder.ACODecoder
    :summary:
    ```
````

### API

`````{py:class} ACODecoder(n_ants: int = 20, n_iterations: int = 1, alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1, use_local_search: bool = True, **kwargs)
:canonical: src.models.subnets.decoders.deepaco.decoder.ACODecoder

Bases: {py:obj}`logic.src.models.common.nonautoregressive_decoder.NonAutoregressiveDecoder`

```{autodoc2-docstring} src.models.subnets.decoders.deepaco.decoder.ACODecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.decoders.deepaco.decoder.ACODecoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, heatmap: torch.Tensor, env: logic.src.envs.base.RL4COEnvBase, **kwargs) -> tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.decoders.deepaco.decoder.ACODecoder.forward

```{autodoc2-docstring} src.models.subnets.decoders.deepaco.decoder.ACODecoder.forward
```

````

````{py:method} construct(td: tensordict.TensorDict, heatmap: torch.Tensor, env: logic.src.envs.base.RL4COEnvBase, num_starts: int = 1, return_all: bool = False, **kwargs) -> typing.Dict[str, torch.Tensor]
:canonical: src.models.subnets.decoders.deepaco.decoder.ACODecoder.construct

```{autodoc2-docstring} src.models.subnets.decoders.deepaco.decoder.ACODecoder.construct
```

````

````{py:method} _run_ants(prob_matrix: torch.Tensor, dist_matrix: torch.Tensor, td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, return_all: bool = False) -> typing.Union[typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor], typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
:canonical: src.models.subnets.decoders.deepaco.decoder.ACODecoder._run_ants

```{autodoc2-docstring} src.models.subnets.decoders.deepaco.decoder.ACODecoder._run_ants
```

````

````{py:method} _construct_tour(prob_matrix: torch.Tensor, dist_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.decoders.deepaco.decoder.ACODecoder._construct_tour

```{autodoc2-docstring} src.models.subnets.decoders.deepaco.decoder.ACODecoder._construct_tour
```

````

````{py:method} _compute_tour_cost(tour: torch.Tensor, dist_matrix: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.decoders.deepaco.decoder.ACODecoder._compute_tour_cost

```{autodoc2-docstring} src.models.subnets.decoders.deepaco.decoder.ACODecoder._compute_tour_cost
```

````

````{py:method} _two_opt(tours: torch.Tensor, dist_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.decoders.deepaco.decoder.ACODecoder._two_opt

```{autodoc2-docstring} src.models.subnets.decoders.deepaco.decoder.ACODecoder._two_opt
```

````

`````
