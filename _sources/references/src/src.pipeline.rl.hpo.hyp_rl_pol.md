# {py:mod}`src.pipeline.rl.hpo.hyp_rl_pol`

```{py:module} src.pipeline.rl.hpo.hyp_rl_pol
```

```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl_pol
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HypRLPolicy <src.pipeline.rl.hpo.hyp_rl_pol.HypRLPolicy>`
  - ```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl_pol.HypRLPolicy
    :summary:
    ```
````

### API

`````{py:class} HypRLPolicy(search_space: typing.Dict[str, src.pipeline.rl.hpo.base.ParamSpec], state_dim: int = 64, hidden_dim: int = 128, n_layers: int = 2, normalization: str = 'layer', activation: str = 'relu', device: typing.Optional[torch.device] = None)
:canonical: src.pipeline.rl.hpo.hyp_rl_pol.HypRLPolicy

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl_pol.HypRLPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl_pol.HypRLPolicy.__init__
```

````{py:method} forward(history: typing.List[typing.Tuple[typing.Dict[str, typing.Any], float]], hidden: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None) -> typing.Tuple[typing.Dict[str, typing.Any], typing.Dict[str, torch.Tensor], typing.Tuple[torch.Tensor, torch.Tensor]]
:canonical: src.pipeline.rl.hpo.hyp_rl_pol.HypRLPolicy.forward

```{autodoc2-docstring} src.pipeline.rl.hpo.hyp_rl_pol.HypRLPolicy.forward
```

````

`````
