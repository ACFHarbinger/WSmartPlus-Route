# {py:mod}`src.utils.tasks.losses`

```{py:module} src.utils.tasks.losses
```

```{autodoc2-docstring} src.utils.tasks.losses
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`problem_symmetricity_loss <src.utils.tasks.losses.problem_symmetricity_loss>`
  - ```{autodoc2-docstring} src.utils.tasks.losses.problem_symmetricity_loss
    :summary:
    ```
* - {py:obj}`solution_symmetricity_loss <src.utils.tasks.losses.solution_symmetricity_loss>`
  - ```{autodoc2-docstring} src.utils.tasks.losses.solution_symmetricity_loss
    :summary:
    ```
* - {py:obj}`invariance_loss <src.utils.tasks.losses.invariance_loss>`
  - ```{autodoc2-docstring} src.utils.tasks.losses.invariance_loss
    :summary:
    ```
````

### API

````{py:function} problem_symmetricity_loss(reward: torch.Tensor, log_likelihood: torch.Tensor, dim: int = 1) -> torch.Tensor
:canonical: src.utils.tasks.losses.problem_symmetricity_loss

```{autodoc2-docstring} src.utils.tasks.losses.problem_symmetricity_loss
```
````

````{py:function} solution_symmetricity_loss(reward: torch.Tensor, log_likelihood: torch.Tensor, dim: int = -1) -> torch.Tensor
:canonical: src.utils.tasks.losses.solution_symmetricity_loss

```{autodoc2-docstring} src.utils.tasks.losses.solution_symmetricity_loss
```
````

````{py:function} invariance_loss(proj_embed: torch.Tensor, num_augment: int) -> torch.Tensor
:canonical: src.utils.tasks.losses.invariance_loss

```{autodoc2-docstring} src.utils.tasks.losses.invariance_loss
```
````
