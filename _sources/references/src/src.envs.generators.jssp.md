# {py:mod}`src.envs.generators.jssp`

```{py:module} src.envs.generators.jssp
```

```{autodoc2-docstring} src.envs.generators.jssp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JSSPGenerator <src.envs.generators.jssp.JSSPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.jssp.JSSPGenerator
    :summary:
    ```
````

### API

`````{py:class} JSSPGenerator(num_jobs: int = 10, num_machines: int = 10, min_duration: int = 1, max_duration: int = 99, duration_distribution: str = 'uniform', device: typing.Union[str, torch.device] = 'cpu', **kwargs: typing.Any)
:canonical: src.envs.generators.jssp.JSSPGenerator

Bases: {py:obj}`src.envs.generators.base.Generator`

```{autodoc2-docstring} src.envs.generators.jssp.JSSPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.jssp.JSSPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.jssp.JSSPGenerator._generate

```{autodoc2-docstring} src.envs.generators.jssp.JSSPGenerator._generate
```

````

`````
