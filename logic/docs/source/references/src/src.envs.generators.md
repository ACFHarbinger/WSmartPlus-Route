# {py:mod}`src.envs.generators`

```{py:module} src.envs.generators
```

```{autodoc2-docstring} src.envs.generators
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Generator <src.envs.generators.Generator>`
  - ```{autodoc2-docstring} src.envs.generators.Generator
    :summary:
    ```
* - {py:obj}`VRPPGenerator <src.envs.generators.VRPPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.VRPPGenerator
    :summary:
    ```
* - {py:obj}`WCVRPGenerator <src.envs.generators.WCVRPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.WCVRPGenerator
    :summary:
    ```
* - {py:obj}`SCWCVRPGenerator <src.envs.generators.SCWCVRPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.SCWCVRPGenerator
    :summary:
    ```
* - {py:obj}`TSPGenerator <src.envs.generators.TSPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.TSPGenerator
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_generator <src.envs.generators.get_generator>`
  - ```{autodoc2-docstring} src.envs.generators.get_generator
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GENERATOR_REGISTRY <src.envs.generators.GENERATOR_REGISTRY>`
  - ```{autodoc2-docstring} src.envs.generators.GENERATOR_REGISTRY
    :summary:
    ```
````

### API

`````{py:class} Generator(num_loc: int = 50, min_loc: float = 0.0, max_loc: float = 1.0, loc_distribution: typing.Union[str, typing.Callable] = 'uniform', device: typing.Union[str, torch.device] = 'cpu', **kwargs: typing.Any)
:canonical: src.envs.generators.Generator

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.envs.generators.Generator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.Generator.__init__
```

````{py:property} kwargs
:canonical: src.envs.generators.Generator.kwargs
:type: dict[str, typing.Any]

```{autodoc2-docstring} src.envs.generators.Generator.kwargs
```

````

````{py:method} to(device: typing.Union[str, torch.device]) -> src.envs.generators.Generator
:canonical: src.envs.generators.Generator.to

```{autodoc2-docstring} src.envs.generators.Generator.to
```

````

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.Generator._generate
:abstractmethod:

```{autodoc2-docstring} src.envs.generators.Generator._generate
```

````

````{py:method} __call__(batch_size: typing.Union[int, list[int], tuple[int, ...]] = 1) -> tensordict.TensorDict
:canonical: src.envs.generators.Generator.__call__

```{autodoc2-docstring} src.envs.generators.Generator.__call__
```

````

````{py:method} _generate_locations(batch_size: tuple[int, ...], num_loc: typing.Optional[int] = None) -> torch.Tensor
:canonical: src.envs.generators.Generator._generate_locations

```{autodoc2-docstring} src.envs.generators.Generator._generate_locations
```

````

````{py:method} _uniform_locations(batch_size: tuple[int, ...], num_loc: int) -> torch.Tensor
:canonical: src.envs.generators.Generator._uniform_locations

```{autodoc2-docstring} src.envs.generators.Generator._uniform_locations
```

````

````{py:method} _normal_locations(batch_size: tuple[int, ...], num_loc: int) -> torch.Tensor
:canonical: src.envs.generators.Generator._normal_locations

```{autodoc2-docstring} src.envs.generators.Generator._normal_locations
```

````

````{py:method} _clustered_locations(batch_size: tuple[int, ...], num_loc: int, num_clusters: int = 3) -> torch.Tensor
:canonical: src.envs.generators.Generator._clustered_locations

```{autodoc2-docstring} src.envs.generators.Generator._clustered_locations
```

````

`````

`````{py:class} VRPPGenerator(num_loc: int = 50, min_loc: float = 0.0, max_loc: float = 1.0, loc_distribution: typing.Union[str, typing.Callable] = 'uniform', min_waste: float = 0.0, max_waste: float = 1.0, waste_distribution: str = 'uniform', min_prize: float = 0.0, max_prize: float = 1.0, prize_distribution: str = 'uniform', capacity: float = 1.0, max_length: typing.Optional[float] = None, depot_type: str = 'center', device: typing.Union[str, torch.device] = 'cpu', **kwargs: typing.Any)
:canonical: src.envs.generators.VRPPGenerator

Bases: {py:obj}`src.envs.generators.Generator`

```{autodoc2-docstring} src.envs.generators.VRPPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.VRPPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.VRPPGenerator._generate

```{autodoc2-docstring} src.envs.generators.VRPPGenerator._generate
```

````

````{py:method} _generate_depot(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.VRPPGenerator._generate_depot

```{autodoc2-docstring} src.envs.generators.VRPPGenerator._generate_depot
```

````

````{py:method} _generate_waste(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.VRPPGenerator._generate_waste

```{autodoc2-docstring} src.envs.generators.VRPPGenerator._generate_waste
```

````

````{py:method} _generate_prize(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.VRPPGenerator._generate_prize

```{autodoc2-docstring} src.envs.generators.VRPPGenerator._generate_prize
```

````

`````

`````{py:class} WCVRPGenerator(num_loc: int = 50, min_loc: float = 0.0, max_loc: float = 1.0, loc_distribution: typing.Union[str, typing.Callable] = 'uniform', min_fill: float = 0.0, max_fill: float = 1.0, fill_distribution: str = 'uniform', capacity: float = 100.0, cost_km: float = 1.0, revenue_kg: float = 0.1625, depot_type: str = 'center', device: typing.Union[str, torch.device] = 'cpu', **kwargs: typing.Any)
:canonical: src.envs.generators.WCVRPGenerator

Bases: {py:obj}`src.envs.generators.Generator`

```{autodoc2-docstring} src.envs.generators.WCVRPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.WCVRPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.WCVRPGenerator._generate

```{autodoc2-docstring} src.envs.generators.WCVRPGenerator._generate
```

````

````{py:method} _generate_depot(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.WCVRPGenerator._generate_depot

```{autodoc2-docstring} src.envs.generators.WCVRPGenerator._generate_depot
```

````

````{py:method} _generate_fill_levels(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.WCVRPGenerator._generate_fill_levels

```{autodoc2-docstring} src.envs.generators.WCVRPGenerator._generate_fill_levels
```

````

`````

`````{py:class} SCWCVRPGenerator(num_loc: int = 50, min_loc: float = 0.0, max_loc: float = 1.0, loc_distribution: typing.Union[str, typing.Callable] = 'uniform', min_fill: float = 0.0, max_fill: float = 1.0, fill_distribution: str = 'uniform', capacity: float = 100.0, cost_km: float = 1.0, revenue_kg: float = 0.1625, depot_type: str = 'center', noise_mean: float = 0.0, noise_variance: float = 0.0, device: typing.Union[str, torch.device] = 'cpu', **kwargs: typing.Any)
:canonical: src.envs.generators.SCWCVRPGenerator

Bases: {py:obj}`src.envs.generators.WCVRPGenerator`

```{autodoc2-docstring} src.envs.generators.SCWCVRPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.SCWCVRPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.SCWCVRPGenerator._generate

```{autodoc2-docstring} src.envs.generators.SCWCVRPGenerator._generate
```

````

`````

`````{py:class} TSPGenerator(num_loc: int = 50, min_loc: float = 0.0, max_loc: float = 1.0, loc_distribution: typing.Union[str, typing.Callable] = 'uniform', device: typing.Union[str, torch.device] = 'cpu', **kwargs: typing.Any)
:canonical: src.envs.generators.TSPGenerator

Bases: {py:obj}`src.envs.generators.Generator`

```{autodoc2-docstring} src.envs.generators.TSPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.TSPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.TSPGenerator._generate

```{autodoc2-docstring} src.envs.generators.TSPGenerator._generate
```

````

`````

````{py:data} GENERATOR_REGISTRY
:canonical: src.envs.generators.GENERATOR_REGISTRY
:type: dict[str, type[src.envs.generators.Generator]]
:value: >
   None

```{autodoc2-docstring} src.envs.generators.GENERATOR_REGISTRY
```

````

````{py:function} get_generator(name: str, **kwargs: typing.Any) -> src.envs.generators.Generator
:canonical: src.envs.generators.get_generator

```{autodoc2-docstring} src.envs.generators.get_generator
```
````
