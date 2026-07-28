# {py:mod}`src.envs.generators.atsp`

```{py:module} src.envs.generators.atsp
```

```{autodoc2-docstring} src.envs.generators.atsp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ATSPGenerator <src.envs.generators.atsp.ATSPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.atsp.ATSPGenerator
    :summary:
    ```
````

### API

`````{py:class} ATSPGenerator(num_loc: int = 10, min_dist: float = 0.0, max_dist: float = 1.0, tmat_class: bool = True, device: typing.Union[str, torch.device] = 'cpu', rng: typing.Optional[typing.Any] = None, generator: typing.Optional[torch.Generator] = None, **kwargs: typing.Any)
:canonical: src.envs.generators.atsp.ATSPGenerator

Bases: {py:obj}`src.envs.generators.base.Generator`

```{autodoc2-docstring} src.envs.generators.atsp.ATSPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.atsp.ATSPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.atsp.ATSPGenerator._generate

```{autodoc2-docstring} src.envs.generators.atsp.ATSPGenerator._generate
```

````

`````
