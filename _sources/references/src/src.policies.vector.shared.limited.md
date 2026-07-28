# {py:mod}`src.policies.vector.shared.limited`

```{py:module} src.policies.vector.shared.limited
```

```{autodoc2-docstring} src.policies.vector.shared.limited
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_split_limited <src.policies.vector.shared.limited.vectorized_split_limited>`
  - ```{autodoc2-docstring} src.policies.vector.shared.limited.vectorized_split_limited
    :summary:
    ```
````

### API

````{py:function} vectorized_split_limited(B: int, N: int, device: torch.device, max_vehicles: int, capacity: float, cum_load: torch.Tensor, cum_dist_pad: torch.Tensor, d_0_i: torch.Tensor, d_i_0: torch.Tensor, giant_tours: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.policies.vector.shared.limited.vectorized_split_limited

```{autodoc2-docstring} src.policies.vector.shared.limited.vectorized_split_limited
```
````
