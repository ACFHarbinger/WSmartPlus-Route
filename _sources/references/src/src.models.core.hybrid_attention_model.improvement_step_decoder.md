# {py:mod}`src.models.core.hybrid_attention_model.improvement_step_decoder`

```{py:module} src.models.core.hybrid_attention_model.improvement_step_decoder
```

```{autodoc2-docstring} src.models.core.hybrid_attention_model.improvement_step_decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImprovementStepDecoder <src.models.core.hybrid_attention_model.improvement_step_decoder.ImprovementStepDecoder>`
  - ```{autodoc2-docstring} src.models.core.hybrid_attention_model.improvement_step_decoder.ImprovementStepDecoder
    :summary:
    ```
````

### API

`````{py:class} ImprovementStepDecoder(embed_dim: int = 128, n_operators: int = 6, hidden_dim: int = 128)
:canonical: src.models.core.hybrid_attention_model.improvement_step_decoder.ImprovementStepDecoder

Bases: {py:obj}`logic.src.models.common.improvement.decoder.ImprovementDecoder`

```{autodoc2-docstring} src.models.core.hybrid_attention_model.improvement_step_decoder.ImprovementStepDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.hybrid_attention_model.improvement_step_decoder.ImprovementStepDecoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, embeddings: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]], env: logic.src.envs.base.RL4COEnvBase, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.core.hybrid_attention_model.improvement_step_decoder.ImprovementStepDecoder.forward

```{autodoc2-docstring} src.models.core.hybrid_attention_model.improvement_step_decoder.ImprovementStepDecoder.forward
```

````

`````
