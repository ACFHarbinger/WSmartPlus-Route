# {py:mod}`src.data.processor.processing`

```{py:module} src.data.processor.processing
```

```{autodoc2-docstring} src.data.processor.processing
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`process_data <src.data.processor.processing.process_data>`
  - ```{autodoc2-docstring} src.data.processor.processing.process_data
    :summary:
    ```
* - {py:obj}`process_coordinates <src.data.processor.processing.process_coordinates>`
  - ```{autodoc2-docstring} src.data.processor.processing.process_coordinates
    :summary:
    ```
* - {py:obj}`process_model_data <src.data.processor.processing.process_model_data>`
  - ```{autodoc2-docstring} src.data.processor.processing.process_model_data
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_mapper <src.data.processor.processing._mapper>`
  - ```{autodoc2-docstring} src.data.processor.processing._mapper
    :summary:
    ```
````

### API

````{py:data} _mapper
:canonical: src.data.processor.processing._mapper
:value: >
   'SimulationDataMapper(...)'

```{autodoc2-docstring} src.data.processor.processing._mapper
```

````

````{py:function} process_data(data, bins_coordinates, depot, indices=None)
:canonical: src.data.processor.processing.process_data

```{autodoc2-docstring} src.data.processor.processing.process_data
```
````

````{py:function} process_coordinates(coords, method, col_names=None)
:canonical: src.data.processor.processing.process_coordinates

```{autodoc2-docstring} src.data.processor.processing.process_coordinates
```
````

````{py:function} process_model_data(coordinates, dist_matrix, device, method, configs, edge_threshold, edge_method, area, waste_type, adj_matrix=None)
:canonical: src.data.processor.processing.process_model_data

```{autodoc2-docstring} src.data.processor.processing.process_model_data
```
````
