# {py:mod}`src.pipeline.rl.hpo.hyp_rl_enc`

```{py:module} src.pipeline.rl.hpo.hyp_rl_enc
```

```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl_enc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperparameterEncoder <src.pipeline.rl.hpo.hyp_rl_enc.HyperparameterEncoder>`
  - ```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl_enc.HyperparameterEncoder
    :summary:
    ```
````

### API

`````{py:class} HyperparameterEncoder(search_space: typing.Dict[str, src.pipeline.rl.hpo.base.ParamSpec], embed_dim: int = 32, normalization: str = 'layer', activation: str = 'relu', device: typing.Optional[torch.device] = None)
:canonical: src.pipeline.rl.hpo.hyp_rl_enc.HyperparameterEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl_enc.HyperparameterEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl_enc.HyperparameterEncoder.__init__
```

````{py:method} forward(config: typing.Dict[str, typing.Any]) -> torch.Tensor
:canonical: src.pipeline.rl.hpo.hyp_rl_enc.HyperparameterEncoder.forward

```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl_enc.HyperparameterEncoder.forward
```

````

`````
