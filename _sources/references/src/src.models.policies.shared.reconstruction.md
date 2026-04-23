# {py:mod}`src.models.policies.shared.reconstruction`

```{py:module} src.models.policies.shared.reconstruction
```

```{autodoc2-docstring} src.models.policies.shared.reconstruction
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`reconstruct_routes <src.models.policies.shared.reconstruction.reconstruct_routes>`
  - ```{autodoc2-docstring} src.models.policies.shared.reconstruction.reconstruct_routes
    :summary:
    ```
* - {py:obj}`reconstruct_limited <src.models.policies.shared.reconstruction.reconstruct_limited>`
  - ```{autodoc2-docstring} src.models.policies.shared.reconstruction.reconstruct_limited
    :summary:
    ```
````

### API

````{py:function} reconstruct_routes(B: int, N: int, giant_tours: torch.Tensor, P: torch.Tensor, costs: torch.Tensor) -> typing.Tuple[typing.List[typing.List[int]], torch.Tensor]
:canonical: src.models.policies.shared.reconstruction.reconstruct_routes

```{autodoc2-docstring} src.models.policies.shared.reconstruction.reconstruct_routes
```
````

````{py:function} reconstruct_limited(B: int, N: int, giant_tours: torch.Tensor, P_k: torch.Tensor, best_k: torch.Tensor, costs: torch.Tensor) -> typing.Tuple[typing.List[typing.List[int]], torch.Tensor]
:canonical: src.models.policies.shared.reconstruction.reconstruct_limited

```{autodoc2-docstring} src.models.policies.shared.reconstruction.reconstruct_limited
```
````
