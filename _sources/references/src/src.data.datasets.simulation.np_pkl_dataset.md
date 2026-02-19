# {py:mod}`src.data.datasets.simulation.np_pkl_dataset`

```{py:module} src.data.datasets.simulation.np_pkl_dataset
```

```{autodoc2-docstring} src.data.datasets.simulation.np_pkl_dataset
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NumpyPickleDataset <src.data.datasets.simulation.np_pkl_dataset.NumpyPickleDataset>`
  - ```{autodoc2-docstring} src.data.datasets.simulation.np_pkl_dataset.NumpyPickleDataset
    :summary:
    ```
````

### API

`````{py:class} NumpyPickleDataset(data: typing.List[typing.Tuple[typing.Any, ...]])
:canonical: src.data.datasets.simulation.np_pkl_dataset.NumpyPickleDataset

Bases: {py:obj}`src.data.datasets.simulation.sim_dataset.SimulationDataset`

```{autodoc2-docstring} src.data.datasets.simulation.np_pkl_dataset.NumpyPickleDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.datasets.simulation.np_pkl_dataset.NumpyPickleDataset.__init__
```

````{py:method} __len__() -> int
:canonical: src.data.datasets.simulation.np_pkl_dataset.NumpyPickleDataset.__len__

```{autodoc2-docstring} src.data.datasets.simulation.np_pkl_dataset.NumpyPickleDataset.__len__
```

````

````{py:method} __getitem__(index: int) -> typing.Dict[str, numpy.ndarray]
:canonical: src.data.datasets.simulation.np_pkl_dataset.NumpyPickleDataset.__getitem__

```{autodoc2-docstring} src.data.datasets.simulation.np_pkl_dataset.NumpyPickleDataset.__getitem__
```

````

````{py:method} load(path: str) -> src.data.datasets.simulation.np_pkl_dataset.NumpyPickleDataset
:canonical: src.data.datasets.simulation.np_pkl_dataset.NumpyPickleDataset.load
:staticmethod:

```{autodoc2-docstring} src.data.datasets.simulation.np_pkl_dataset.NumpyPickleDataset.load
```

````

`````
