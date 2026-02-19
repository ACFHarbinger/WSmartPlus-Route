# {py:mod}`src.data.datasets.simulation.pd_xlsx_dataset`

```{py:module} src.data.datasets.simulation.pd_xlsx_dataset
```

```{autodoc2-docstring} src.data.datasets.simulation.pd_xlsx_dataset
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PandasExcelDataset <src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset>`
  - ```{autodoc2-docstring} src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset
    :summary:
    ```
````

### API

`````{py:class} PandasExcelDataset(samples: typing.List[typing.Dict[str, numpy.ndarray]])
:canonical: src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset

Bases: {py:obj}`src.data.datasets.simulation.sim_dataset.SimulationDataset`

```{autodoc2-docstring} src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset.__init__
```

````{py:method} __len__() -> int
:canonical: src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset.__len__

```{autodoc2-docstring} src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset.__len__
```

````

````{py:method} __getitem__(index: int) -> typing.Dict[str, numpy.ndarray]
:canonical: src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset.__getitem__

```{autodoc2-docstring} src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset.__getitem__
```

````

````{py:method} load(path: str) -> src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset
:canonical: src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset.load
:staticmethod:

```{autodoc2-docstring} src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset.load
```

````

````{py:method} _parse_sheet(df: pandas.DataFrame) -> typing.Dict[str, numpy.ndarray]
:canonical: src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset._parse_sheet
:staticmethod:

```{autodoc2-docstring} src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset._parse_sheet
```

````

````{py:method} save(path: str) -> None
:canonical: src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset.save

```{autodoc2-docstring} src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset.save
```

````

````{py:method} _sample_to_dataframe(sample: typing.Dict[str, numpy.ndarray]) -> pandas.DataFrame
:canonical: src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset._sample_to_dataframe
:staticmethod:

```{autodoc2-docstring} src.data.datasets.simulation.pd_xlsx_dataset.PandasExcelDataset._sample_to_dataframe
```

````

`````
