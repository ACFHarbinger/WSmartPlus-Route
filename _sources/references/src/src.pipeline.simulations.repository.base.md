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

````{py:method} get_indices(filename, n_samples, n_nodes, data_size, lock=None)
:canonical: src.pipeline.simulations.repository.base.SimulationRepository.get_indices
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.repository.base.SimulationRepository.get_indices
```

````

````{py:method} get_depot(area, data_dir=None)
:canonical: src.pipeline.simulations.repository.base.SimulationRepository.get_depot
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.repository.base.SimulationRepository.get_depot
```

````

````{py:method} get_simulator_data(number_of_bins, area='Rio Maior', waste_type=None, lock=None, data_dir=None)
:canonical: src.pipeline.simulations.repository.base.SimulationRepository.get_simulator_data
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.repository.base.SimulationRepository.get_simulator_data
```

````

````{py:method} get_area_params(area, waste_type)
:canonical: src.pipeline.simulations.repository.base.SimulationRepository.get_area_params
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.repository.base.SimulationRepository.get_area_params
```

````

`````
