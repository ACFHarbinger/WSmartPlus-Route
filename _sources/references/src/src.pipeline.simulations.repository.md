# {py:mod}`src.pipeline.simulations.repository`

```{py:module} src.pipeline.simulations.repository
```

```{autodoc2-docstring} src.pipeline.simulations.repository
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.pipeline.simulations.repository.filesystem
src.pipeline.simulations.repository.npz
src.pipeline.simulations.repository.xlsx
src.pipeline.simulations.repository.csv
src.pipeline.simulations.repository.base
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`set_repository <src.pipeline.simulations.repository.set_repository>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository.set_repository
    :summary:
    ```
* - {py:obj}`load_indices <src.pipeline.simulations.repository.load_indices>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository.load_indices
    :summary:
    ```
* - {py:obj}`load_depot <src.pipeline.simulations.repository.load_depot>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository.load_depot
    :summary:
    ```
* - {py:obj}`load_simulator_data <src.pipeline.simulations.repository.load_simulator_data>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository.load_simulator_data
    :summary:
    ```
* - {py:obj}`load_area_and_waste_type_params <src.pipeline.simulations.repository.load_area_and_waste_type_params>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository.load_area_and_waste_type_params
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_REPOSITORY <src.pipeline.simulations.repository._REPOSITORY>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository._REPOSITORY
    :summary:
    ```
* - {py:obj}`__all__ <src.pipeline.simulations.repository.__all__>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository.__all__
    :summary:
    ```
````

### API

````{py:data} _REPOSITORY
:canonical: src.pipeline.simulations.repository._REPOSITORY
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.repository._REPOSITORY
```

````

````{py:function} set_repository(repo: typing.Union[src.pipeline.simulations.repository.filesystem.FileSystemRepository, src.pipeline.simulations.repository.npz.NumpyDictRepository, src.pipeline.simulations.repository.xlsx.PandasExcelRepository, src.pipeline.simulations.repository.csv.PandasCsvRepository]) -> None
:canonical: src.pipeline.simulations.repository.set_repository

```{autodoc2-docstring} src.pipeline.simulations.repository.set_repository
```
````

````{py:function} load_indices(filename, n_samples, n_nodes, data_size, lock=None)
:canonical: src.pipeline.simulations.repository.load_indices

```{autodoc2-docstring} src.pipeline.simulations.repository.load_indices
```
````

````{py:function} load_depot(data_dir, area='Rio Maior')
:canonical: src.pipeline.simulations.repository.load_depot

```{autodoc2-docstring} src.pipeline.simulations.repository.load_depot
```
````

````{py:function} load_simulator_data(data_dir, number_of_bins, area='Rio Maior', waste_type=None, lock=None)
:canonical: src.pipeline.simulations.repository.load_simulator_data

```{autodoc2-docstring} src.pipeline.simulations.repository.load_simulator_data
```
````

````{py:function} load_area_and_waste_type_params(area, waste_type)
:canonical: src.pipeline.simulations.repository.load_area_and_waste_type_params

```{autodoc2-docstring} src.pipeline.simulations.repository.load_area_and_waste_type_params
```
````

````{py:data} __all__
:canonical: src.pipeline.simulations.repository.__all__
:value: >
   ['SimulationRepository', 'FileSystemRepository', 'NumpyDictRepository', 'PandasExcelRepository', 'Pa...

```{autodoc2-docstring} src.pipeline.simulations.repository.__all__
```

````
