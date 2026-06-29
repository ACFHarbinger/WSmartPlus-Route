# {py:mod}`src.policies.vector.shared.linear`

```{py:module} src.policies.vector.shared.linear
```

```{autodoc2-docstring} src.policies.vector.shared.linear
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_linear_split <src.policies.vector.shared.linear.vectorized_linear_split>`
  - ```{autodoc2-docstring} src.policies.vector.shared.linear.vectorized_linear_split
    :summary:
    ```
````

### API

````{py:function} vectorized_linear_split(giant_tours: torch.Tensor, dist_matrix: torch.Tensor, wastes: torch.Tensor, vehicle_capacity: typing.Union[float, torch.Tensor], max_len: typing.Optional[int] = None, max_vehicles: typing.Optional[int] = None) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.policies.vector.shared.linear.vectorized_linear_split

```{autodoc2-docstring} src.policies.vector.shared.linear.vectorized_linear_split
```
````
