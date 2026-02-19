# {py:mod}`src.pipeline.rl.hpo.dehb`

```{py:module} src.pipeline.rl.hpo.dehb
```

```{autodoc2-docstring} src.pipeline.rl.hpo.dehb
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DifferentialEvolutionHyperband <src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband>`
  - ```{autodoc2-docstring} src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband
    :summary:
    ```
````

### API

`````{py:class} DifferentialEvolutionHyperband(cfg: logic.src.configs.Config, objective_fn: typing.Callable, search_space: typing.Optional[typing.Dict[str, src.pipeline.rl.hpo.base.ParamSpec]] = None, min_fidelity: int = 1, max_fidelity: int = 10, eta: int = 3, n_workers: int = 1, output_path: str = './dehb_output', **kwargs: typing.Any)
:canonical: src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband

Bases: {py:obj}`src.pipeline.rl.hpo.base.BaseHPO`

```{autodoc2-docstring} src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband.__init__
```

````{py:method} run() -> float
:canonical: src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband.run

```{autodoc2-docstring} src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband.run
```

````

`````
