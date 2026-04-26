# {py:mod}`src.models.core.hybrid_attention_model.hybrid_two_step_policy`

```{py:module} src.models.core.hybrid_attention_model.hybrid_two_step_policy
```

```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_two_step_policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HybridTwoStagePolicy <src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy>`
  - ```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy
    :summary:
    ```
````

### API

`````{py:class} HybridTwoStagePolicy(env_name: str, embed_dim: int = 128, hidden_dim: int = 128, n_encode_layers: int = 3, n_heads: int = 8, refine_steps: int = 10, seed: int = 42, **kwargs: typing.Any)
:canonical: src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy

Bases: {py:obj}`logic.src.models.common.autoregressive.policy.AutoregressivePolicy`

```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy.__init__
```

````{py:method} __getstate__() -> typing.Dict[str, typing.Any]
:canonical: src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy.__getstate__

```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy.__getstate__
```

````

````{py:method} __setstate__(state: typing.Dict[str, typing.Any]) -> None
:canonical: src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy.__setstate__

```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy.__setstate__
```

````

````{py:method} _initialize_tours(td: tensordict.TensorDict, strategy: str, embeddings: torch.Tensor, env: typing.Optional[logic.src.envs.base.base.RL4COEnvBase] = None, **kwargs: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy._initialize_tours

```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy._initialize_tours
```

````

````{py:method} _apply_operator_step(td: tensordict.TensorDict, embeddings: torch.Tensor, current_tours: torch.Tensor, dist_matrix_all: torch.Tensor, removed_nodes_state: torch.Tensor, strategy: str, env: typing.Optional[logic.src.envs.base.base.RL4COEnvBase] = None) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy._apply_operator_step

```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy._apply_operator_step
```

````

````{py:method} _execute_refinement_operator(operator_fn: typing.Any, sub_tours: torch.Tensor, sub_dist: torch.Tensor, sub_removed: torch.Tensor, device: torch.device) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy._execute_refinement_operator

```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy._execute_refinement_operator
```

````

````{py:method} _assemble_removed_state(removed_list: list, device: torch.device) -> torch.Tensor
:canonical: src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy._assemble_removed_state

```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy._assemble_removed_state
```

````

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.base.RL4COEnvBase] = None, strategy: str = 'greedy', **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy.forward

```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy.forward
```

````

````{py:method} _get_dist_matrix(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy._get_dist_matrix

```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy._get_dist_matrix
```

````

````{py:method} _get_random_tours(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy._get_random_tours

```{autodoc2-docstring} src.models.core.hybrid_attention_model.hybrid_two_step_policy.HybridTwoStagePolicy._get_random_tours
```

````

`````
