# {py:mod}`src.pipeline.simulations.repository.xlsx`

```{py:module} src.pipeline.simulations.repository.xlsx
```

```{autodoc2-docstring} src.pipeline.simulations.repository.xlsx
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PandasExcelRepository <src.pipeline.simulations.repository.xlsx.PandasExcelRepository>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository.xlsx.PandasExcelRepository
    :summary:
    ```
````

### API

`````{py:class} PandasExcelRepository(dataset: logic.src.data.datasets.PandasExcelDataset, sample_id: int = 0)
:canonical: src.pipeline.simulations.repository.xlsx.PandasExcelRepository

Bases: {py:obj}`src.pipeline.simulations.repository.base.SimulationRepository`

```{autodoc2-docstring} src.pipeline.simulations.repository.xlsx.PandasExcelRepository
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.repository.xlsx.PandasExcelRepository.__init__
```

````{py:method} set_sample(sample_id: int) -> None
:canonical: src.pipeline.simulations.repository.xlsx.PandasExcelRepository.set_sample

```{autodoc2-docstring} src.pipeline.simulations.repository.xlsx.PandasExcelRepository.set_sample
```

````

````{py:method} get_indices(filename, n_samples, n_nodes, data_size, lock=None)
:canonical: src.pipeline.simulations.repository.xlsx.PandasExcelRepository.get_indices

```{autodoc2-docstring} src.pipeline.simulations.repository.xlsx.PandasExcelRepository.get_indices
```

````

````{py:method} get_depot(area=None, data_dir=None)
:canonical: src.pipeline.simulations.repository.xlsx.PandasExcelRepository.get_depot

```{autodoc2-docstring} src.pipeline.simulations.repository.xlsx.PandasExcelRepository.get_depot
```

````

````{py:method} get_simulator_data(number_of_bins, area='Rio Maior', waste_type=None, lock=None, data_dir=None)
:canonical: src.pipeline.simulations.repository.xlsx.PandasExcelRepository.get_simulator_data

```{autodoc2-docstring} src.pipeline.simulations.repository.xlsx.PandasExcelRepository.get_simulator_data
```

````

````{py:method} get_area_params(area, waste_type)
:canonical: src.pipeline.simulations.repository.xlsx.PandasExcelRepository.get_area_params

```{autodoc2-docstring} src.pipeline.simulations.repository.xlsx.PandasExcelRepository.get_area_params
```

````

`````
