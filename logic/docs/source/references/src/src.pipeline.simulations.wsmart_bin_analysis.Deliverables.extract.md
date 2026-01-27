# {py:mod}`src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract`

```{py:module} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`pre_process_data <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract.pre_process_data>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract.pre_process_data
    :summary:
    ```
* - {py:obj}`import_separate_file <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract.import_separate_file>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract.import_separate_file
    :summary:
    ```
* - {py:obj}`import_same_file <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract.import_same_file>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract.import_same_file
    :summary:
    ```
* - {py:obj}`container_global_sorted_wrapper <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract.container_global_sorted_wrapper>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract.container_global_sorted_wrapper
    :summary:
    ```
````

### API

````{py:function} pre_process_data(df_fill: pandas.DataFrame, df_collection: pandas.DataFrame, id_header_fill: str, date_header_fill: str, date_format_fill: str, fill_header_fill: str, id_header_collect: str, date_header_collect: str, date_format_collect: str, start_date: str = '01/01/2020', end_date: str = '01/01/2025') -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract.pre_process_data

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract.pre_process_data
```
````

````{py:function} import_separate_file(src_fill: list[str], src_collect: list[str], sep_f: str = ',', sep_c: str = ',', path: str = '', print_firt_line=True) -> tuple[pandas.DataFrame, pandas.DataFrame]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract.import_separate_file

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract.import_separate_file
```
````

````{py:function} import_same_file(src_fill: str, collect_id_header: str, sep: str = ',', path: str = '', print_first_line=True) -> tuple[pandas.DataFrame, pandas.DataFrame]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract.import_same_file

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract.import_same_file
```
````

````{py:function} container_global_sorted_wrapper(fill_: pandas.DataFrame, collect_: pandas.DataFrame, info: pandas.DataFrame) -> tuple[dict[int, src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container], list[int]]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract.container_global_sorted_wrapper

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.extract.container_global_sorted_wrapper
```
````
