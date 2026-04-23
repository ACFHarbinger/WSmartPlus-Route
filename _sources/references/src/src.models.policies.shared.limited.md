# {py:mod}`src.models.policies.shared.limited`

```{py:module} src.models.policies.shared.limited
```

```{autodoc2-docstring} src.models.policies.shared.limited
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_split_limited <src.models.policies.shared.limited.vectorized_split_limited>`
  - ```{autodoc2-docstring} src.models.policies.shared.limited.vectorized_split_limited
    :summary:
    ```
````

### API

````{py:function} vectorized_split_limited(B: int, N: int, device: torch.device, max_vehicles: int, capacity: float, cum_load: torch.Tensor, cum_dist_pad: torch.Tensor, d_0_i: torch.Tensor, d_i_0: torch.Tensor, giant_tours: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.policies.shared.limited.vectorized_split_limited

```{autodoc2-docstring} src.models.policies.shared.limited.vectorized_split_limited
```
````
