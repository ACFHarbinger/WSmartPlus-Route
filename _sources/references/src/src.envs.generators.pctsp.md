# {py:mod}`src.envs.generators.pctsp`

```{py:module} src.envs.generators.pctsp
```

```{autodoc2-docstring} src.envs.generators.pctsp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PCTSPGenerator <src.envs.generators.pctsp.PCTSPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.pctsp.PCTSPGenerator
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MAX_PENALTIES <src.envs.generators.pctsp.MAX_PENALTIES>`
  - ```{autodoc2-docstring} src.envs.generators.pctsp.MAX_PENALTIES
    :summary:
    ```
````

### API

````{py:data} MAX_PENALTIES
:canonical: src.envs.generators.pctsp.MAX_PENALTIES
:type: dict[int, float]
:value: >
   None

```{autodoc2-docstring} src.envs.generators.pctsp.MAX_PENALTIES
```

````

`````{py:class} PCTSPGenerator(num_loc: int = 20, min_loc: float = 0.0, max_loc: float = 1.0, loc_distribution: typing.Union[str, typing.Callable] = 'uniform', penalty_factor: float = 3.0, prize_required: float = 1.0, depot_type: str = 'random', device: typing.Union[str, torch.device] = 'cpu', rng: typing.Optional[typing.Any] = None, generator: typing.Optional[torch.Generator] = None, **kwargs: typing.Any)
:canonical: src.envs.generators.pctsp.PCTSPGenerator

Bases: {py:obj}`src.envs.generators.base.Generator`

```{autodoc2-docstring} src.envs.generators.pctsp.PCTSPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.pctsp.PCTSPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.pctsp.PCTSPGenerator._generate

```{autodoc2-docstring} src.envs.generators.pctsp.PCTSPGenerator._generate
```

````

````{py:method} _generate_depot(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.pctsp.PCTSPGenerator._generate_depot

```{autodoc2-docstring} src.envs.generators.pctsp.PCTSPGenerator._generate_depot
```

````

`````
