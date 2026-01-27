# {py:mod}`src.pipeline.simulations.loader`

```{py:module} src.pipeline.simulations.loader
```

```{autodoc2-docstring} src.pipeline.simulations.loader
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationRepository <src.pipeline.simulations.loader.SimulationRepository>`
  - ```{autodoc2-docstring} src.pipeline.simulations.loader.SimulationRepository
    :summary:
    ```
* - {py:obj}`FileSystemRepository <src.pipeline.simulations.loader.FileSystemRepository>`
  - ```{autodoc2-docstring} src.pipeline.simulations.loader.FileSystemRepository
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`load_indices <src.pipeline.simulations.loader.load_indices>`
  - ```{autodoc2-docstring} src.pipeline.simulations.loader.load_indices
    :summary:
    ```
* - {py:obj}`load_depot <src.pipeline.simulations.loader.load_depot>`
  - ```{autodoc2-docstring} src.pipeline.simulations.loader.load_depot
    :summary:
    ```
* - {py:obj}`load_simulator_data <src.pipeline.simulations.loader.load_simulator_data>`
  - ```{autodoc2-docstring} src.pipeline.simulations.loader.load_simulator_data
    :summary:
    ```
* - {py:obj}`load_area_and_waste_type_params <src.pipeline.simulations.loader.load_area_and_waste_type_params>`
  - ```{autodoc2-docstring} src.pipeline.simulations.loader.load_area_and_waste_type_params
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_repository <src.pipeline.simulations.loader._repository>`
  - ```{autodoc2-docstring} src.pipeline.simulations.loader._repository
    :summary:
    ```
````

### API

`````{py:class} SimulationRepository
:canonical: src.pipeline.simulations.loader.SimulationRepository

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.simulations.loader.SimulationRepository
```

````{py:method} get_indices(filename, n_samples, n_nodes, data_size, lock=None)
:canonical: src.pipeline.simulations.loader.SimulationRepository.get_indices
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.loader.SimulationRepository.get_indices
```

````

````{py:method} get_depot(area, data_dir=None)
:canonical: src.pipeline.simulations.loader.SimulationRepository.get_depot
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.loader.SimulationRepository.get_depot
```

````

````{py:method} get_simulator_data(number_of_bins, area='Rio Maior', waste_type=None, lock=None, data_dir=None)
:canonical: src.pipeline.simulations.loader.SimulationRepository.get_simulator_data
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.loader.SimulationRepository.get_simulator_data
```

````

````{py:method} get_area_params(area, waste_type)
:canonical: src.pipeline.simulations.loader.SimulationRepository.get_area_params
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.loader.SimulationRepository.get_area_params
```

````

`````

`````{py:class} FileSystemRepository(data_root_dir)
:canonical: src.pipeline.simulations.loader.FileSystemRepository

Bases: {py:obj}`src.pipeline.simulations.loader.SimulationRepository`

```{autodoc2-docstring} src.pipeline.simulations.loader.FileSystemRepository
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.loader.FileSystemRepository.__init__
```

````{py:method} _get_data_dir(override_dir=None)
:canonical: src.pipeline.simulations.loader.FileSystemRepository._get_data_dir

```{autodoc2-docstring} src.pipeline.simulations.loader.FileSystemRepository._get_data_dir
```

````

````{py:method} get_indices(filename, n_samples, n_nodes, data_size, lock=None)
:canonical: src.pipeline.simulations.loader.FileSystemRepository.get_indices

```{autodoc2-docstring} src.pipeline.simulations.loader.FileSystemRepository.get_indices
```

````

````{py:method} get_depot(area, data_dir=None)
:canonical: src.pipeline.simulations.loader.FileSystemRepository.get_depot

```{autodoc2-docstring} src.pipeline.simulations.loader.FileSystemRepository.get_depot
```

````

````{py:method} get_simulator_data(number_of_bins, area='Rio Maior', waste_type=None, lock=None, data_dir=None)
:canonical: src.pipeline.simulations.loader.FileSystemRepository.get_simulator_data

```{autodoc2-docstring} src.pipeline.simulations.loader.FileSystemRepository.get_simulator_data
```

````

````{py:method} get_area_params(area, waste_type)
:canonical: src.pipeline.simulations.loader.FileSystemRepository.get_area_params

```{autodoc2-docstring} src.pipeline.simulations.loader.FileSystemRepository.get_area_params
```

````

`````

````{py:data} _repository
:canonical: src.pipeline.simulations.loader._repository
:value: >
   'FileSystemRepository(...)'

```{autodoc2-docstring} src.pipeline.simulations.loader._repository
```

````

````{py:function} load_indices(filename, n_samples, n_nodes, data_size, lock=None)
:canonical: src.pipeline.simulations.loader.load_indices

```{autodoc2-docstring} src.pipeline.simulations.loader.load_indices
```
````

````{py:function} load_depot(data_dir, area='Rio Maior')
:canonical: src.pipeline.simulations.loader.load_depot

```{autodoc2-docstring} src.pipeline.simulations.loader.load_depot
```
````

````{py:function} load_simulator_data(data_dir, number_of_bins, area='Rio Maior', waste_type=None, lock=None)
:canonical: src.pipeline.simulations.loader.load_simulator_data

```{autodoc2-docstring} src.pipeline.simulations.loader.load_simulator_data
```
````

````{py:function} load_area_and_waste_type_params(area, waste_type)
:canonical: src.pipeline.simulations.loader.load_area_and_waste_type_params

```{autodoc2-docstring} src.pipeline.simulations.loader.load_area_and_waste_type_params
```
````
