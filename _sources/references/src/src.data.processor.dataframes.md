# {py:mod}`src.data.processor.dataframes`

```{py:module} src.data.processor.dataframes
```

```{autodoc2-docstring} src.data.processor.dataframes
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`sort_dataframe <src.data.processor.dataframes.sort_dataframe>`
  - ```{autodoc2-docstring} src.data.processor.dataframes.sort_dataframe
    :summary:
    ```
* - {py:obj}`get_df_types <src.data.processor.dataframes.get_df_types>`
  - ```{autodoc2-docstring} src.data.processor.dataframes.get_df_types
    :summary:
    ```
* - {py:obj}`setup_df <src.data.processor.dataframes.setup_df>`
  - ```{autodoc2-docstring} src.data.processor.dataframes.setup_df
    :summary:
    ```
* - {py:obj}`sample_df <src.data.processor.dataframes.sample_df>`
  - ```{autodoc2-docstring} src.data.processor.dataframes.sample_df
    :summary:
    ```
* - {py:obj}`process_indices <src.data.processor.dataframes.process_indices>`
  - ```{autodoc2-docstring} src.data.processor.dataframes.process_indices
    :summary:
    ```
* - {py:obj}`create_dataframe_from_matrix <src.data.processor.dataframes.create_dataframe_from_matrix>`
  - ```{autodoc2-docstring} src.data.processor.dataframes.create_dataframe_from_matrix
    :summary:
    ```
* - {py:obj}`convert_to_dict <src.data.processor.dataframes.convert_to_dict>`
  - ```{autodoc2-docstring} src.data.processor.dataframes.convert_to_dict
    :summary:
    ```
* - {py:obj}`save_matrix_to_excel <src.data.processor.dataframes.save_matrix_to_excel>`
  - ```{autodoc2-docstring} src.data.processor.dataframes.save_matrix_to_excel
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_mapper <src.data.processor.dataframes._mapper>`
  - ```{autodoc2-docstring} src.data.processor.dataframes._mapper
    :summary:
    ```
````

### API

````{py:data} _mapper
:canonical: src.data.processor.dataframes._mapper
:value: >
   'SimulationDataMapper(...)'

```{autodoc2-docstring} src.data.processor.dataframes._mapper
```

````

````{py:function} sort_dataframe(df, metric_tosort, ascending_order=True)
:canonical: src.data.processor.dataframes.sort_dataframe

```{autodoc2-docstring} src.data.processor.dataframes.sort_dataframe
```
````

````{py:function} get_df_types(df, prec='32')
:canonical: src.data.processor.dataframes.get_df_types

```{autodoc2-docstring} src.data.processor.dataframes.get_df_types
```
````

````{py:function} setup_df(depot, df, col_names, index_name='#bin')
:canonical: src.data.processor.dataframes.setup_df

```{autodoc2-docstring} src.data.processor.dataframes.setup_df
```
````

````{py:function} sample_df(df, n_elems, depot=None, output_path=None)
:canonical: src.data.processor.dataframes.sample_df

```{autodoc2-docstring} src.data.processor.dataframes.sample_df
```
````

````{py:function} process_indices(df, indices)
:canonical: src.data.processor.dataframes.process_indices

```{autodoc2-docstring} src.data.processor.dataframes.process_indices
```
````

````{py:function} create_dataframe_from_matrix(matrix)
:canonical: src.data.processor.dataframes.create_dataframe_from_matrix

```{autodoc2-docstring} src.data.processor.dataframes.create_dataframe_from_matrix
```
````

````{py:function} convert_to_dict(bins_coordinates)
:canonical: src.data.processor.dataframes.convert_to_dict

```{autodoc2-docstring} src.data.processor.dataframes.convert_to_dict
```
````

````{py:function} save_matrix_to_excel(matrix, results_dir, seed, data_dist, policy, sample_id)
:canonical: src.data.processor.dataframes.save_matrix_to_excel

```{autodoc2-docstring} src.data.processor.dataframes.save_matrix_to_excel
```
````
