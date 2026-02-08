# {py:mod}`src.envs.generators.tsp`

```{py:module} src.envs.generators.tsp
```

```{autodoc2-docstring} src.envs.generators.tsp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TSPGenerator <src.envs.generators.tsp.TSPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.tsp.TSPGenerator
    :summary:
    ```
````

### API

`````{py:class} TSPGenerator(num_loc: int = 50, min_loc: float = 0.0, max_loc: float = 1.0, loc_distribution: typing.Union[str, typing.Callable] = 'uniform', device: typing.Union[str, torch.device] = 'cpu', **kwargs: typing.Any)
:canonical: src.envs.generators.tsp.TSPGenerator

Bases: {py:obj}`src.envs.generators.base.Generator`

```{autodoc2-docstring} src.envs.generators.tsp.TSPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.tsp.TSPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.tsp.TSPGenerator._generate

```{autodoc2-docstring} src.envs.generators.tsp.TSPGenerator._generate
```

````

`````
