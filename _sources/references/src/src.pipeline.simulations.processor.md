# {py:mod}`src.pipeline.simulations.processor`

```{py:module} src.pipeline.simulations.processor
```

```{autodoc2-docstring} src.pipeline.simulations.processor
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationDataMapper <src.pipeline.simulations.processor.SimulationDataMapper>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor.SimulationDataMapper
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`sort_dataframe <src.pipeline.simulations.processor.sort_dataframe>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor.sort_dataframe
    :summary:
    ```
* - {py:obj}`get_df_types <src.pipeline.simulations.processor.get_df_types>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor.get_df_types
    :summary:
    ```
* - {py:obj}`setup_df <src.pipeline.simulations.processor.setup_df>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor.setup_df
    :summary:
    ```
* - {py:obj}`sample_df <src.pipeline.simulations.processor.sample_df>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor.sample_df
    :summary:
    ```
* - {py:obj}`process_indices <src.pipeline.simulations.processor.process_indices>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor.process_indices
    :summary:
    ```
* - {py:obj}`process_data <src.pipeline.simulations.processor.process_data>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor.process_data
    :summary:
    ```
* - {py:obj}`haversine_distance_old <src.pipeline.simulations.processor.haversine_distance_old>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor.haversine_distance_old
    :summary:
    ```
* - {py:obj}`process_coordinates <src.pipeline.simulations.processor.process_coordinates>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor.process_coordinates
    :summary:
    ```
* - {py:obj}`process_model_data <src.pipeline.simulations.processor.process_model_data>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor.process_model_data
    :summary:
    ```
* - {py:obj}`create_dataframe_from_matrix <src.pipeline.simulations.processor.create_dataframe_from_matrix>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor.create_dataframe_from_matrix
    :summary:
    ```
* - {py:obj}`convert_to_dict <src.pipeline.simulations.processor.convert_to_dict>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor.convert_to_dict
    :summary:
    ```
* - {py:obj}`save_matrix_to_excel <src.pipeline.simulations.processor.save_matrix_to_excel>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor.save_matrix_to_excel
    :summary:
    ```
* - {py:obj}`setup_basedata <src.pipeline.simulations.processor.setup_basedata>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor.setup_basedata
    :summary:
    ```
* - {py:obj}`setup_dist_path_tup <src.pipeline.simulations.processor.setup_dist_path_tup>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor.setup_dist_path_tup
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_mapper <src.pipeline.simulations.processor._mapper>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor._mapper
    :summary:
    ```
````

### API

`````{py:class} SimulationDataMapper
:canonical: src.pipeline.simulations.processor.SimulationDataMapper

```{autodoc2-docstring} src.pipeline.simulations.processor.SimulationDataMapper
```

````{py:method} sort_dataframe(df: pandas.DataFrame, metric_tosort: str, ascending_order: bool = True) -> pandas.DataFrame
:canonical: src.pipeline.simulations.processor.SimulationDataMapper.sort_dataframe

```{autodoc2-docstring} src.pipeline.simulations.processor.SimulationDataMapper.sort_dataframe
```

````

````{py:method} get_df_types(df: pandas.DataFrame, prec: str = '32') -> typing.Dict[str, str]
:canonical: src.pipeline.simulations.processor.SimulationDataMapper.get_df_types

```{autodoc2-docstring} src.pipeline.simulations.processor.SimulationDataMapper.get_df_types
```

````

````{py:method} setup_df(depot: pandas.DataFrame, df: pandas.DataFrame, col_names: typing.List[str], index_name: typing.Optional[str] = '#bin') -> pandas.DataFrame
:canonical: src.pipeline.simulations.processor.SimulationDataMapper.setup_df

```{autodoc2-docstring} src.pipeline.simulations.processor.SimulationDataMapper.setup_df
```

````

````{py:method} sample_df(df: pandas.DataFrame, n_elems: int, depot: typing.Optional[pandas.DataFrame] = None, output_path: typing.Optional[str] = None) -> pandas.DataFrame
:canonical: src.pipeline.simulations.processor.SimulationDataMapper.sample_df

```{autodoc2-docstring} src.pipeline.simulations.processor.SimulationDataMapper.sample_df
```

````

````{py:method} process_indices(df: pandas.DataFrame, indices: typing.Optional[typing.List[int]]) -> pandas.DataFrame
:canonical: src.pipeline.simulations.processor.SimulationDataMapper.process_indices

```{autodoc2-docstring} src.pipeline.simulations.processor.SimulationDataMapper.process_indices
```

````

````{py:method} process_raw_data(data: pandas.DataFrame, bins_coordinates: pandas.DataFrame, depot: pandas.DataFrame, indices: typing.Optional[typing.List[int]] = None) -> typing.Tuple[pandas.DataFrame, pandas.DataFrame]
:canonical: src.pipeline.simulations.processor.SimulationDataMapper.process_raw_data

```{autodoc2-docstring} src.pipeline.simulations.processor.SimulationDataMapper.process_raw_data
```

````

````{py:method} format_coordinates(coords: typing.Any, method: str, col_names: typing.Optional[typing.List[str]] = ['Lat', 'Lng']) -> typing.Tuple[typing.Any, typing.Any]
:canonical: src.pipeline.simulations.processor.SimulationDataMapper.format_coordinates

```{autodoc2-docstring} src.pipeline.simulations.processor.SimulationDataMapper.format_coordinates
```

````

````{py:method} process_model_input(coordinates, dist_matrix, device, method, configs, edge_threshold, edge_method, area, waste_type, adj_matrix=None)
:canonical: src.pipeline.simulations.processor.SimulationDataMapper.process_model_input

```{autodoc2-docstring} src.pipeline.simulations.processor.SimulationDataMapper.process_model_input
```

````

````{py:method} save_results(matrix, results_dir, seed, data_dist, policy, sample_id)
:canonical: src.pipeline.simulations.processor.SimulationDataMapper.save_results

```{autodoc2-docstring} src.pipeline.simulations.processor.SimulationDataMapper.save_results
```

````

`````

````{py:data} _mapper
:canonical: src.pipeline.simulations.processor._mapper
:value: >
   'SimulationDataMapper(...)'

```{autodoc2-docstring} src.pipeline.simulations.processor._mapper
```

````

````{py:function} sort_dataframe(df, metric_tosort, ascending_order=True)
:canonical: src.pipeline.simulations.processor.sort_dataframe

```{autodoc2-docstring} src.pipeline.simulations.processor.sort_dataframe
```
````

````{py:function} get_df_types(df, prec='32')
:canonical: src.pipeline.simulations.processor.get_df_types

```{autodoc2-docstring} src.pipeline.simulations.processor.get_df_types
```
````

````{py:function} setup_df(depot, df, col_names, index_name='#bin')
:canonical: src.pipeline.simulations.processor.setup_df

```{autodoc2-docstring} src.pipeline.simulations.processor.setup_df
```
````

````{py:function} sample_df(df, n_elems, depot=None, output_path=None)
:canonical: src.pipeline.simulations.processor.sample_df

```{autodoc2-docstring} src.pipeline.simulations.processor.sample_df
```
````

````{py:function} process_indices(df, indices)
:canonical: src.pipeline.simulations.processor.process_indices

```{autodoc2-docstring} src.pipeline.simulations.processor.process_indices
```
````

````{py:function} process_data(data, bins_coordinates, depot, indices=None)
:canonical: src.pipeline.simulations.processor.process_data

```{autodoc2-docstring} src.pipeline.simulations.processor.process_data
```
````

````{py:function} haversine_distance_old(lat1, lng1, lat2, lng2)
:canonical: src.pipeline.simulations.processor.haversine_distance_old

```{autodoc2-docstring} src.pipeline.simulations.processor.haversine_distance_old
```
````

````{py:function} process_coordinates(coords, method, col_names=['Lat', 'Lng'])
:canonical: src.pipeline.simulations.processor.process_coordinates

```{autodoc2-docstring} src.pipeline.simulations.processor.process_coordinates
```
````

````{py:function} process_model_data(coordinates, dist_matrix, device, method, configs, edge_threshold, edge_method, area, waste_type, adj_matrix=None)
:canonical: src.pipeline.simulations.processor.process_model_data

```{autodoc2-docstring} src.pipeline.simulations.processor.process_model_data
```
````

````{py:function} create_dataframe_from_matrix(matrix)
:canonical: src.pipeline.simulations.processor.create_dataframe_from_matrix

```{autodoc2-docstring} src.pipeline.simulations.processor.create_dataframe_from_matrix
```
````

````{py:function} convert_to_dict(bins_coordinates)
:canonical: src.pipeline.simulations.processor.convert_to_dict

```{autodoc2-docstring} src.pipeline.simulations.processor.convert_to_dict
```
````

````{py:function} save_matrix_to_excel(matrix, results_dir, seed, data_dist, policy, sample_id)
:canonical: src.pipeline.simulations.processor.save_matrix_to_excel

```{autodoc2-docstring} src.pipeline.simulations.processor.save_matrix_to_excel
```
````

````{py:function} setup_basedata(n_bins, data_dir, area, waste_type)
:canonical: src.pipeline.simulations.processor.setup_basedata

```{autodoc2-docstring} src.pipeline.simulations.processor.setup_basedata
```
````

````{py:function} setup_dist_path_tup(bins_coordinates, size, dist_method, dm_filepath, env_filename, gapik_file, symkey_name, device, edge_thresh, edge_method, focus_idx=None)
:canonical: src.pipeline.simulations.processor.setup_dist_path_tup

```{autodoc2-docstring} src.pipeline.simulations.processor.setup_dist_path_tup
```
````
