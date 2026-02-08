# {py:mod}`src.envs.generators.scwcvrp`

```{py:module} src.envs.generators.scwcvrp
```

```{autodoc2-docstring} src.envs.generators.scwcvrp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SCWCVRPGenerator <src.envs.generators.scwcvrp.SCWCVRPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.scwcvrp.SCWCVRPGenerator
    :summary:
    ```
````

### API

`````{py:class} SCWCVRPGenerator(num_loc: int = 50, min_loc: float = 0.0, max_loc: float = 1.0, loc_distribution: typing.Union[str, typing.Callable] = 'uniform', min_fill: float = 0.0, max_fill: float = 1.0, fill_distribution: str = 'uniform', capacity: float = 100.0, cost_km: float = 1.0, revenue_kg: float = 0.1625, depot_type: str = 'center', noise_mean: float = 0.0, noise_variance: float = 0.0, device: typing.Union[str, torch.device] = 'cpu', **kwargs: typing.Any)
:canonical: src.envs.generators.scwcvrp.SCWCVRPGenerator

Bases: {py:obj}`src.envs.generators.wcvrp.WCVRPGenerator`

```{autodoc2-docstring} src.envs.generators.scwcvrp.SCWCVRPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.scwcvrp.SCWCVRPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.scwcvrp.SCWCVRPGenerator._generate

```{autodoc2-docstring} src.envs.generators.scwcvrp.SCWCVRPGenerator._generate
```

````

`````
