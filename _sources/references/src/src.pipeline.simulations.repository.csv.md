# {py:mod}`src.pipeline.simulations.repository.csv`

```{py:module} src.pipeline.simulations.repository.csv
```

```{autodoc2-docstring} src.pipeline.simulations.repository.csv
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PandasCsvRepository <src.pipeline.simulations.repository.csv.PandasCsvRepository>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository.csv.PandasCsvRepository
    :summary:
    ```
````

### API

`````{py:class} PandasCsvRepository(dataset: logic.src.data.datasets.PandasCsvDataset, sample_id: int = 0)
:canonical: src.pipeline.simulations.repository.csv.PandasCsvRepository

Bases: {py:obj}`src.pipeline.simulations.repository.base.SimulationRepository`

```{autodoc2-docstring} src.pipeline.simulations.repository.csv.PandasCsvRepository
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.repository.csv.PandasCsvRepository.__init__
```

````{py:method} set_sample(sample_id: int) -> None
:canonical: src.pipeline.simulations.repository.csv.PandasCsvRepository.set_sample

```{autodoc2-docstring} src.pipeline.simulations.repository.csv.PandasCsvRepository.set_sample
```

````

````{py:method} get_indices(filename, n_samples, n_nodes, data_size, lock=None)
:canonical: src.pipeline.simulations.repository.csv.PandasCsvRepository.get_indices

```{autodoc2-docstring} src.pipeline.simulations.repository.csv.PandasCsvRepository.get_indices
```

````

````{py:method} get_depot(area=None, data_dir=None)
:canonical: src.pipeline.simulations.repository.csv.PandasCsvRepository.get_depot

```{autodoc2-docstring} src.pipeline.simulations.repository.csv.PandasCsvRepository.get_depot
```

````

````{py:method} get_simulator_data(number_of_bins, area='Rio Maior', waste_type=None, lock=None, data_dir=None)
:canonical: src.pipeline.simulations.repository.csv.PandasCsvRepository.get_simulator_data

```{autodoc2-docstring} src.pipeline.simulations.repository.csv.PandasCsvRepository.get_simulator_data
```

````

````{py:method} get_area_params(area, waste_type)
:canonical: src.pipeline.simulations.repository.csv.PandasCsvRepository.get_area_params

```{autodoc2-docstring} src.pipeline.simulations.repository.csv.PandasCsvRepository.get_area_params
```

````

`````
