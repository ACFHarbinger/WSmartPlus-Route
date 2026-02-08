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
src.pipeline.simulations.repository.base
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

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

* - {py:obj}`_repository <src.pipeline.simulations.repository._repository>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository._repository
    :summary:
    ```
* - {py:obj}`__all__ <src.pipeline.simulations.repository.__all__>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository.__all__
    :summary:
    ```
````

### API

````{py:data} _repository
:canonical: src.pipeline.simulations.repository._repository
:value: >
   'FileSystemRepository(...)'

```{autodoc2-docstring} src.pipeline.simulations.repository._repository
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
   ['SimulationRepository', 'FileSystemRepository', 'load_indices', 'load_depot', 'load_simulator_data'...

```{autodoc2-docstring} src.pipeline.simulations.repository.__all__
```

````
