# {py:mod}`src.pipeline.simulations.processor.mapper`

```{py:module} src.pipeline.simulations.processor.mapper
```

```{autodoc2-docstring} src.pipeline.simulations.processor.mapper
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationDataMapper <src.pipeline.simulations.processor.mapper.SimulationDataMapper>`
  - ```{autodoc2-docstring} src.pipeline.simulations.processor.mapper.SimulationDataMapper
    :summary:
    ```
````

### API

`````{py:class} SimulationDataMapper
:canonical: src.pipeline.simulations.processor.mapper.SimulationDataMapper

```{autodoc2-docstring} src.pipeline.simulations.processor.mapper.SimulationDataMapper
```

````{py:method} sort_dataframe(df: pandas.DataFrame, metric_tosort: str, ascending_order: bool = True) -> pandas.DataFrame
:canonical: src.pipeline.simulations.processor.mapper.SimulationDataMapper.sort_dataframe

```{autodoc2-docstring} src.pipeline.simulations.processor.mapper.SimulationDataMapper.sort_dataframe
```

````

````{py:method} get_df_types(df: pandas.DataFrame, prec: str = '32') -> typing.Dict[str, str]
:canonical: src.pipeline.simulations.processor.mapper.SimulationDataMapper.get_df_types

```{autodoc2-docstring} src.pipeline.simulations.processor.mapper.SimulationDataMapper.get_df_types
```

````

````{py:method} setup_df(depot: pandas.DataFrame, df: pandas.DataFrame, col_names: typing.List[str], index_name: typing.Optional[str] = '#bin') -> pandas.DataFrame
:canonical: src.pipeline.simulations.processor.mapper.SimulationDataMapper.setup_df

```{autodoc2-docstring} src.pipeline.simulations.processor.mapper.SimulationDataMapper.setup_df
```

````

````{py:method} sample_df(df: pandas.DataFrame, n_elems: int, depot: typing.Optional[pandas.DataFrame] = None, output_path: typing.Optional[str] = None) -> pandas.DataFrame
:canonical: src.pipeline.simulations.processor.mapper.SimulationDataMapper.sample_df

```{autodoc2-docstring} src.pipeline.simulations.processor.mapper.SimulationDataMapper.sample_df
```

````

````{py:method} process_indices(df: pandas.DataFrame, indices: typing.Optional[typing.List[int]]) -> pandas.DataFrame
:canonical: src.pipeline.simulations.processor.mapper.SimulationDataMapper.process_indices

```{autodoc2-docstring} src.pipeline.simulations.processor.mapper.SimulationDataMapper.process_indices
```

````

````{py:method} process_raw_data(data: pandas.DataFrame, bins_coordinates: pandas.DataFrame, depot: pandas.DataFrame, indices: typing.Optional[typing.List[int]] = None) -> typing.Tuple[pandas.DataFrame, pandas.DataFrame]
:canonical: src.pipeline.simulations.processor.mapper.SimulationDataMapper.process_raw_data

```{autodoc2-docstring} src.pipeline.simulations.processor.mapper.SimulationDataMapper.process_raw_data
```

````

````{py:method} process_model_input(coordinates, dist_matrix, device, method, configs, edge_threshold, edge_method, area, waste_type, adj_matrix=None)
:canonical: src.pipeline.simulations.processor.mapper.SimulationDataMapper.process_model_input

```{autodoc2-docstring} src.pipeline.simulations.processor.mapper.SimulationDataMapper.process_model_input
```

````

````{py:method} save_results(matrix, results_dir, seed, data_dist, policy, sample_id)
:canonical: src.pipeline.simulations.processor.mapper.SimulationDataMapper.save_results

```{autodoc2-docstring} src.pipeline.simulations.processor.mapper.SimulationDataMapper.save_results
```

````

`````
