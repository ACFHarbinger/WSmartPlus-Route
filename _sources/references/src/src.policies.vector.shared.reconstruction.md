# {py:mod}`src.policies.vector.shared.reconstruction`

```{py:module} src.policies.vector.shared.reconstruction
```

```{autodoc2-docstring} src.policies.vector.shared.reconstruction
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`reconstruct_routes <src.policies.vector.shared.reconstruction.reconstruct_routes>`
  - ```{autodoc2-docstring} src.policies.vector.shared.reconstruction.reconstruct_routes
    :summary:
    ```
* - {py:obj}`reconstruct_limited <src.policies.vector.shared.reconstruction.reconstruct_limited>`
  - ```{autodoc2-docstring} src.policies.vector.shared.reconstruction.reconstruct_limited
    :summary:
    ```
````

### API

````{py:function} reconstruct_routes(B: int, N: int, giant_tours: torch.Tensor, P: torch.Tensor, costs: torch.Tensor) -> typing.Tuple[typing.List[typing.List[int]], torch.Tensor]
:canonical: src.policies.vector.shared.reconstruction.reconstruct_routes

```{autodoc2-docstring} src.policies.vector.shared.reconstruction.reconstruct_routes
```
````

````{py:function} reconstruct_limited(B: int, N: int, giant_tours: torch.Tensor, P_k: torch.Tensor, best_k: torch.Tensor, costs: torch.Tensor) -> typing.Tuple[typing.List[typing.List[int]], torch.Tensor]
:canonical: src.policies.vector.shared.reconstruction.reconstruct_limited

```{autodoc2-docstring} src.policies.vector.shared.reconstruction.reconstruct_limited
```
````
