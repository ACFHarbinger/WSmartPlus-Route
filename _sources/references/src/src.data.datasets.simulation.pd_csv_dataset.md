# {py:mod}`src.data.datasets.simulation.pd_csv_dataset`

```{py:module} src.data.datasets.simulation.pd_csv_dataset
```

```{autodoc2-docstring} src.data.datasets.simulation.pd_csv_dataset
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PandasCsvDataset <src.data.datasets.simulation.pd_csv_dataset.PandasCsvDataset>`
  - ```{autodoc2-docstring} src.data.datasets.simulation.pd_csv_dataset.PandasCsvDataset
    :summary:
    ```
````

### API

`````{py:class} PandasCsvDataset(sample: typing.Dict[str, typing.Any])
:canonical: src.data.datasets.simulation.pd_csv_dataset.PandasCsvDataset

Bases: {py:obj}`src.data.datasets.simulation.sim_dataset.SimulationDataset`

```{autodoc2-docstring} src.data.datasets.simulation.pd_csv_dataset.PandasCsvDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.datasets.simulation.pd_csv_dataset.PandasCsvDataset.__init__
```

````{py:method} __len__() -> int
:canonical: src.data.datasets.simulation.pd_csv_dataset.PandasCsvDataset.__len__

```{autodoc2-docstring} src.data.datasets.simulation.pd_csv_dataset.PandasCsvDataset.__len__
```

````

````{py:method} __getitem__(index: int) -> typing.Dict[str, typing.Any]
:canonical: src.data.datasets.simulation.pd_csv_dataset.PandasCsvDataset.__getitem__

```{autodoc2-docstring} src.data.datasets.simulation.pd_csv_dataset.PandasCsvDataset.__getitem__
```

````

````{py:method} load(path: str, area: typing.Optional[str] = None, waste_type: typing.Optional[str] = None) -> src.data.datasets.simulation.pd_csv_dataset.PandasCsvDataset
:canonical: src.data.datasets.simulation.pd_csv_dataset.PandasCsvDataset.load
:staticmethod:

```{autodoc2-docstring} src.data.datasets.simulation.pd_csv_dataset.PandasCsvDataset.load
```

````

````{py:method} _parse_df(df: pandas.DataFrame, area: typing.Optional[str] = None, waste_type: typing.Optional[str] = None) -> typing.Dict[str, typing.Any]
:canonical: src.data.datasets.simulation.pd_csv_dataset.PandasCsvDataset._parse_df
:staticmethod:

```{autodoc2-docstring} src.data.datasets.simulation.pd_csv_dataset.PandasCsvDataset._parse_df
```

````

`````
