# {py:mod}`src.pipeline.rl.core.losses.kl_divergence_loss`

```{py:module} src.pipeline.rl.core.losses.kl_divergence_loss
```

```{autodoc2-docstring} src.pipeline.rl.core.losses.kl_divergence_loss
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`kl_divergence_loss <src.pipeline.rl.core.losses.kl_divergence_loss.kl_divergence_loss>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.losses.kl_divergence_loss.kl_divergence_loss
    :summary:
    ```
* - {py:obj}`reverse_kl_divergence_loss <src.pipeline.rl.core.losses.kl_divergence_loss.reverse_kl_divergence_loss>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.losses.kl_divergence_loss.reverse_kl_divergence_loss
    :summary:
    ```
````

### API

````{py:function} kl_divergence_loss(log_probs: torch.Tensor, target_log_probs: torch.Tensor, reduction: str = 'mean') -> torch.Tensor
:canonical: src.pipeline.rl.core.losses.kl_divergence_loss.kl_divergence_loss

```{autodoc2-docstring} src.pipeline.rl.core.losses.kl_divergence_loss.kl_divergence_loss
```
````

````{py:function} reverse_kl_divergence_loss(log_probs: torch.Tensor, target_log_probs: torch.Tensor, reduction: str = 'mean') -> torch.Tensor
:canonical: src.pipeline.rl.core.losses.kl_divergence_loss.reverse_kl_divergence_loss

```{autodoc2-docstring} src.pipeline.rl.core.losses.kl_divergence_loss.reverse_kl_divergence_loss
```
````
