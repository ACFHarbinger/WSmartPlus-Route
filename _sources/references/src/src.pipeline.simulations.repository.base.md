# {py:mod}`src.pipeline.simulations.repository.base`

```{py:module} src.pipeline.simulations.repository.base
```

```{autodoc2-docstring} src.pipeline.simulations.repository.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationRepository <src.pipeline.simulations.repository.base.SimulationRepository>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository.base.SimulationRepository
    :summary:
    ```
````

### API

`````{py:class} SimulationRepository
:canonical: src.pipeline.simulations.repository.base.SimulationRepository

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.simulations.repository.base.SimulationRepository
```

````{py:method} get_indices(filename: typing.Any, n_samples: int, n_nodes: int, data_size: int, lock: typing.Optional[typing.Any] = None) -> typing.List[typing.List[int]]
:canonical: src.pipeline.simulations.repository.base.SimulationRepository.get_indices
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.repository.base.SimulationRepository.get_indices
```

````

````{py:method} get_depot(area: typing.Any, data_dir: typing.Optional[str] = None) -> pandas.DataFrame
:canonical: src.pipeline.simulations.repository.base.SimulationRepository.get_depot
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.repository.base.SimulationRepository.get_depot
```

````

````{py:method} get_simulator_data(number_of_bins: int, area: str = 'Rio Maior', waste_type: typing.Optional[str] = None, lock: typing.Optional[typing.Any] = None, data_dir: typing.Optional[str] = None) -> typing.Tuple[pandas.DataFrame, pandas.DataFrame]
:canonical: src.pipeline.simulations.repository.base.SimulationRepository.get_simulator_data
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.repository.base.SimulationRepository.get_simulator_data
```

````

````{py:method} get_area_params(area: typing.Any, waste_type: typing.Any) -> typing.Tuple[float, float, float, float, float]
:canonical: src.pipeline.simulations.repository.base.SimulationRepository.get_area_params
:staticmethod:

```{autodoc2-docstring} src.pipeline.simulations.repository.base.SimulationRepository.get_area_params
```

````

`````
