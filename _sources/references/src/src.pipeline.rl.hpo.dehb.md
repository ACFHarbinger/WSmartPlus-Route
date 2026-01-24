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

`````{py:class} DifferentialEvolutionHyperband(cs: typing.Dict[str, typing.Tuple[float, float]], f: typing.Callable, min_fidelity: int = 1, max_fidelity: int = 10, eta: int = 3, n_workers: int = 1, output_path: str = './dehb_output', **kwargs)
:canonical: src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband

```{autodoc2-docstring} src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband.__init__
```

````{py:method} _sample_random_config() -> numpy.ndarray
:canonical: src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband._sample_random_config

```{autodoc2-docstring} src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband._sample_random_config
```

````

````{py:method} _init_population(size: int)
:canonical: src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband._init_population

```{autodoc2-docstring} src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband._init_population
```

````

````{py:method} _eval(config_vec: numpy.ndarray, fidelity: int) -> float
:canonical: src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband._eval

```{autodoc2-docstring} src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband._eval
```

````

````{py:method} run(fevals: int = 100, **kwargs)
:canonical: src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband.run

```{autodoc2-docstring} src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband.run
```

````

````{py:method} get_incumbents()
:canonical: src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband.get_incumbents

```{autodoc2-docstring} src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband.get_incumbents
```

````

`````
