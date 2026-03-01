# {py:mod}`src.data.processor.mapper`

```{py:module} src.data.processor.mapper
```

```{autodoc2-docstring} src.data.processor.mapper
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationDataMapper <src.data.processor.mapper.SimulationDataMapper>`
  - ```{autodoc2-docstring} src.data.processor.mapper.SimulationDataMapper
    :summary:
    ```
````

### API

`````{py:class} SimulationDataMapper
:canonical: src.data.processor.mapper.SimulationDataMapper

```{autodoc2-docstring} src.data.processor.mapper.SimulationDataMapper
```

````{py:method} sort_dataframe(df: pandas.DataFrame, metric_tosort: str, ascending_order: bool = True) -> pandas.DataFrame
:canonical: src.data.processor.mapper.SimulationDataMapper.sort_dataframe

```{autodoc2-docstring} src.data.processor.mapper.SimulationDataMapper.sort_dataframe
```

````

````{py:method} get_df_types(df: pandas.DataFrame, prec: str = '32') -> typing.Dict[str, str]
:canonical: src.data.processor.mapper.SimulationDataMapper.get_df_types

```{autodoc2-docstring} src.data.processor.mapper.SimulationDataMapper.get_df_types
```

````

````{py:method} setup_df(depot: pandas.DataFrame, df: pandas.DataFrame, col_names: typing.List[str], index_name: typing.Optional[str] = '#bin') -> pandas.DataFrame
:canonical: src.data.processor.mapper.SimulationDataMapper.setup_df

```{autodoc2-docstring} src.data.processor.mapper.SimulationDataMapper.setup_df
```

````

````{py:method} sample_df(df: pandas.DataFrame, n_elems: int, depot: typing.Optional[pandas.DataFrame] = None, output_path: typing.Optional[str] = None) -> pandas.DataFrame
:canonical: src.data.processor.mapper.SimulationDataMapper.sample_df

```{autodoc2-docstring} src.data.processor.mapper.SimulationDataMapper.sample_df
```

````

````{py:method} process_indices(df: pandas.DataFrame, indices: typing.Optional[typing.List[int]]) -> pandas.DataFrame
:canonical: src.data.processor.mapper.SimulationDataMapper.process_indices

```{autodoc2-docstring} src.data.processor.mapper.SimulationDataMapper.process_indices
```

````

````{py:method} process_raw_data(data: pandas.DataFrame, bins_coordinates: pandas.DataFrame, depot: pandas.DataFrame, indices: typing.Optional[typing.List[int]] = None) -> typing.Tuple[pandas.DataFrame, pandas.DataFrame]
:canonical: src.data.processor.mapper.SimulationDataMapper.process_raw_data

```{autodoc2-docstring} src.data.processor.mapper.SimulationDataMapper.process_raw_data
```

````

````{py:method} _prepare_model_data(coordinates: typing.Any, method: str, configs: typing.Dict[str, typing.Any], problem_size: int) -> typing.Dict[str, torch.Tensor]
:canonical: src.data.processor.mapper.SimulationDataMapper._prepare_model_data

```{autodoc2-docstring} src.data.processor.mapper.SimulationDataMapper._prepare_model_data
```

````

````{py:method} _prepare_edges(dist_matrix: numpy.ndarray, configs: typing.Dict[str, typing.Any], device: torch.device, edge_threshold: float, edge_method: str, adj_matrix: typing.Optional[numpy.ndarray]) -> typing.Optional[torch.Tensor]
:canonical: src.data.processor.mapper.SimulationDataMapper._prepare_edges

```{autodoc2-docstring} src.data.processor.mapper.SimulationDataMapper._prepare_edges
```

````

````{py:method} _load_profit_vars(area: str, waste_type: str) -> typing.Dict[str, float]
:canonical: src.data.processor.mapper.SimulationDataMapper._load_profit_vars

```{autodoc2-docstring} src.data.processor.mapper.SimulationDataMapper._load_profit_vars
```

````

````{py:method} process_model_input(coordinates: typing.Any, dist_matrix: numpy.ndarray, device: torch.device, method: str, configs: typing.Dict[str, typing.Any], edge_threshold: float, edge_method: str, area: str, waste_type: str, adj_matrix: typing.Optional[numpy.ndarray] = None) -> typing.Tuple[typing.Dict[str, torch.Tensor], typing.Tuple[typing.Optional[torch.Tensor], torch.Tensor], typing.Dict[str, float]]
:canonical: src.data.processor.mapper.SimulationDataMapper.process_model_input

```{autodoc2-docstring} src.data.processor.mapper.SimulationDataMapper.process_model_input
```

````

````{py:method} save_results(matrix: typing.List[typing.List[float]], results_dir: str, seed: int, data_dist: str, policy: str, sample_id: int) -> None
:canonical: src.data.processor.mapper.SimulationDataMapper.save_results

```{autodoc2-docstring} src.data.processor.mapper.SimulationDataMapper.save_results
```

````

`````
