# {py:mod}`src.pipeline.simulations.repository.filesystem`

```{py:module} src.pipeline.simulations.repository.filesystem
```

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FileSystemRepository <src.pipeline.simulations.repository.filesystem.FileSystemRepository>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository
    :summary:
    ```
````

### API

`````{py:class} FileSystemRepository(data_root_dir)
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository

Bases: {py:obj}`src.pipeline.simulations.repository.base.SimulationRepository`

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository.__init__
```

````{py:method} _get_data_dir(override_dir=None)
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository._get_data_dir

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository._get_data_dir
```

````

````{py:method} get_indices(filename, n_samples, n_nodes, data_size, lock=None)
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository.get_indices

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository.get_indices
```

````

````{py:method} get_depot(area, data_dir=None)
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository.get_depot

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository.get_depot
```

````

````{py:method} get_simulator_data(number_of_bins, area='Rio Maior', waste_type=None, lock=None, data_dir=None)
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository.get_simulator_data

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository.get_simulator_data
```

````

````{py:method} get_area_params(area, waste_type)
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository.get_area_params

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository.get_area_params
```

````

`````
