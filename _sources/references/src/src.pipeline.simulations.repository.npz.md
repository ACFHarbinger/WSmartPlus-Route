# {py:mod}`src.pipeline.simulations.repository.npz`

```{py:module} src.pipeline.simulations.repository.npz
```

```{autodoc2-docstring} src.pipeline.simulations.repository.npz
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NumpyDictRepository <src.pipeline.simulations.repository.npz.NumpyDictRepository>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository.npz.NumpyDictRepository
    :summary:
    ```
````

### API

`````{py:class} NumpyDictRepository(dataset: logic.src.data.datasets.NumpyDictDataset, sample_id: int = 0)
:canonical: src.pipeline.simulations.repository.npz.NumpyDictRepository

Bases: {py:obj}`src.pipeline.simulations.repository.base.SimulationRepository`

```{autodoc2-docstring} src.pipeline.simulations.repository.npz.NumpyDictRepository
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.repository.npz.NumpyDictRepository.__init__
```

````{py:method} set_sample(sample_id: int) -> None
:canonical: src.pipeline.simulations.repository.npz.NumpyDictRepository.set_sample

```{autodoc2-docstring} src.pipeline.simulations.repository.npz.NumpyDictRepository.set_sample
```

````

````{py:method} get_indices(filename, n_samples, n_nodes, data_size, lock=None)
:canonical: src.pipeline.simulations.repository.npz.NumpyDictRepository.get_indices

```{autodoc2-docstring} src.pipeline.simulations.repository.npz.NumpyDictRepository.get_indices
```

````

````{py:method} get_depot(area=None, data_dir=None)
:canonical: src.pipeline.simulations.repository.npz.NumpyDictRepository.get_depot

```{autodoc2-docstring} src.pipeline.simulations.repository.npz.NumpyDictRepository.get_depot
```

````

````{py:method} get_simulator_data(number_of_bins, area='Rio Maior', waste_type=None, lock=None, data_dir=None)
:canonical: src.pipeline.simulations.repository.npz.NumpyDictRepository.get_simulator_data

```{autodoc2-docstring} src.pipeline.simulations.repository.npz.NumpyDictRepository.get_simulator_data
```

````

````{py:method} get_area_params(area, waste_type)
:canonical: src.pipeline.simulations.repository.npz.NumpyDictRepository.get_area_params

```{autodoc2-docstring} src.pipeline.simulations.repository.npz.NumpyDictRepository.get_area_params
```

````

`````
