# {py:mod}`src.models.policies.shared.linear`

```{py:module} src.models.policies.shared.linear
```

```{autodoc2-docstring} src.models.policies.shared.linear
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_linear_split <src.models.policies.shared.linear.vectorized_linear_split>`
  - ```{autodoc2-docstring} src.models.policies.shared.linear.vectorized_linear_split
    :summary:
    ```
````

### API

````{py:function} vectorized_linear_split(giant_tours: torch.Tensor, dist_matrix: torch.Tensor, wastes: torch.Tensor, vehicle_capacity: typing.Union[float, torch.Tensor], max_len: typing.Optional[int] = None, max_vehicles: typing.Optional[int] = None) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.policies.shared.linear.vectorized_linear_split

```{autodoc2-docstring} src.models.policies.shared.linear.vectorized_linear_split
```
````
