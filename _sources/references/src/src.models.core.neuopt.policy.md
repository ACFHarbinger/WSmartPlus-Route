# {py:mod}`src.models.core.neuopt.policy`

```{py:module} src.models.core.neuopt.policy
```

```{autodoc2-docstring} src.models.core.neuopt.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuOptPolicy <src.models.core.neuopt.policy.NeuOptPolicy>`
  - ```{autodoc2-docstring} src.models.core.neuopt.policy.NeuOptPolicy
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`execute_neuopt_basis_sequence <src.models.core.neuopt.policy.execute_neuopt_basis_sequence>`
  - ```{autodoc2-docstring} src.models.core.neuopt.policy.execute_neuopt_basis_sequence
    :summary:
    ```
````

### API

````{py:function} execute_neuopt_basis_sequence(solution: torch.Tensor, actions: torch.Tensor) -> torch.Tensor
:canonical: src.models.core.neuopt.policy.execute_neuopt_basis_sequence

```{autodoc2-docstring} src.models.core.neuopt.policy.execute_neuopt_basis_sequence
```
````

`````{py:class} NeuOptPolicy(embed_dim: int = 128, num_heads: int = 8, num_layers: int = 3, **kwargs: typing.Any)
:canonical: src.models.core.neuopt.policy.NeuOptPolicy

Bases: {py:obj}`logic.src.models.common.improvement.policy.ImprovementPolicy`

```{autodoc2-docstring} src.models.core.neuopt.policy.NeuOptPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.neuopt.policy.NeuOptPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Any = None, strategy: str = 'greedy', num_starts: int = 1, max_steps: int | None = None, phase: str = 'train', return_actions: bool = True, k_basis: int | None = None, **kwargs: typing.Any) -> dict[str, typing.Any]
:canonical: src.models.core.neuopt.policy.NeuOptPolicy.forward

```{autodoc2-docstring} src.models.core.neuopt.policy.NeuOptPolicy.forward
```

````

`````
