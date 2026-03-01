# {py:mod}`src.pipeline.simulations.repository.dataset`

```{py:module} src.pipeline.simulations.repository.dataset
```

```{autodoc2-docstring} src.pipeline.simulations.repository.dataset
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DatasetRepository <src.pipeline.simulations.repository.dataset.DatasetRepository>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository.dataset.DatasetRepository
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_DatasetType <src.pipeline.simulations.repository.dataset._DatasetType>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository.dataset._DatasetType
    :summary:
    ```
````

### API

````{py:data} _DatasetType
:canonical: src.pipeline.simulations.repository.dataset._DatasetType
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.repository.dataset._DatasetType
```

````

`````{py:class} DatasetRepository(dataset: src.pipeline.simulations.repository.dataset._DatasetType, sample_id: int = 0)
:canonical: src.pipeline.simulations.repository.dataset.DatasetRepository

Bases: {py:obj}`src.pipeline.simulations.repository.base.SimulationRepository`

```{autodoc2-docstring} src.pipeline.simulations.repository.dataset.DatasetRepository
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.repository.dataset.DatasetRepository.__init__
```

````{py:method} set_sample(sample_id: int) -> None
:canonical: src.pipeline.simulations.repository.dataset.DatasetRepository.set_sample

```{autodoc2-docstring} src.pipeline.simulations.repository.dataset.DatasetRepository.set_sample
```

````

````{py:method} get_indices(filename: typing.Any, n_samples: int, n_nodes: int, data_size: int, lock: typing.Optional[typing.Any] = None) -> typing.List[typing.List[int]]
:canonical: src.pipeline.simulations.repository.dataset.DatasetRepository.get_indices

```{autodoc2-docstring} src.pipeline.simulations.repository.dataset.DatasetRepository.get_indices
```

````

````{py:method} get_depot(area: typing.Any, data_dir: typing.Optional[str] = None) -> pandas.DataFrame
:canonical: src.pipeline.simulations.repository.dataset.DatasetRepository.get_depot

```{autodoc2-docstring} src.pipeline.simulations.repository.dataset.DatasetRepository.get_depot
```

````

````{py:method} get_simulator_data(number_of_bins: int, area: str = 'Rio Maior', waste_type: typing.Optional[str] = None, lock: typing.Optional[typing.Any] = None, data_dir: typing.Optional[str] = None) -> typing.Tuple[pandas.DataFrame, pandas.DataFrame]
:canonical: src.pipeline.simulations.repository.dataset.DatasetRepository.get_simulator_data

```{autodoc2-docstring} src.pipeline.simulations.repository.dataset.DatasetRepository.get_simulator_data
```

````

`````
