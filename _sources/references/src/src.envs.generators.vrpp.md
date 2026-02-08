# {py:mod}`src.envs.generators.vrpp`

```{py:module} src.envs.generators.vrpp
```

```{autodoc2-docstring} src.envs.generators.vrpp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VRPPGenerator <src.envs.generators.vrpp.VRPPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.vrpp.VRPPGenerator
    :summary:
    ```
````

### API

`````{py:class} VRPPGenerator(num_loc: int = 50, min_loc: float = 0.0, max_loc: float = 1.0, loc_distribution: typing.Union[str, typing.Callable] = 'uniform', min_waste: float = 0.0, max_waste: float = 1.0, waste_distribution: str = 'uniform', min_prize: float = 0.0, max_prize: float = 1.0, prize_distribution: str = 'uniform', capacity: float = 1.0, max_length: typing.Optional[float] = None, depot_type: str = 'center', device: typing.Union[str, torch.device] = 'cpu', **kwargs: typing.Any)
:canonical: src.envs.generators.vrpp.VRPPGenerator

Bases: {py:obj}`src.envs.generators.base.Generator`

```{autodoc2-docstring} src.envs.generators.vrpp.VRPPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.vrpp.VRPPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.vrpp.VRPPGenerator._generate

```{autodoc2-docstring} src.envs.generators.vrpp.VRPPGenerator._generate
```

````

````{py:method} _generate_depot(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.vrpp.VRPPGenerator._generate_depot

```{autodoc2-docstring} src.envs.generators.vrpp.VRPPGenerator._generate_depot
```

````

````{py:method} _generate_waste(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.vrpp.VRPPGenerator._generate_waste

```{autodoc2-docstring} src.envs.generators.vrpp.VRPPGenerator._generate_waste
```

````

````{py:method} _generate_prize(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.vrpp.VRPPGenerator._generate_prize

```{autodoc2-docstring} src.envs.generators.vrpp.VRPPGenerator._generate_prize
```

````

`````
