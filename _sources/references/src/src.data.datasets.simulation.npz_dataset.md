# {py:mod}`src.data.datasets.simulation.npz_dataset`

```{py:module} src.data.datasets.simulation.npz_dataset
```

```{autodoc2-docstring} src.data.datasets.simulation.npz_dataset
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NumpyDictDataset <src.data.datasets.simulation.npz_dataset.NumpyDictDataset>`
  - ```{autodoc2-docstring} src.data.datasets.simulation.npz_dataset.NumpyDictDataset
    :summary:
    ```
````

### API

`````{py:class} NumpyDictDataset(data: typing.Dict[str, numpy.ndarray])
:canonical: src.data.datasets.simulation.npz_dataset.NumpyDictDataset

Bases: {py:obj}`src.data.datasets.simulation.sim_dataset.SimulationDataset`

```{autodoc2-docstring} src.data.datasets.simulation.npz_dataset.NumpyDictDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.datasets.simulation.npz_dataset.NumpyDictDataset.__init__
```

````{py:method} __len__() -> int
:canonical: src.data.datasets.simulation.npz_dataset.NumpyDictDataset.__len__

```{autodoc2-docstring} src.data.datasets.simulation.npz_dataset.NumpyDictDataset.__len__
```

````

````{py:method} __getitem__(index: int) -> typing.Dict[str, numpy.ndarray]
:canonical: src.data.datasets.simulation.npz_dataset.NumpyDictDataset.__getitem__

```{autodoc2-docstring} src.data.datasets.simulation.npz_dataset.NumpyDictDataset.__getitem__
```

````

````{py:method} load(path: str) -> src.data.datasets.simulation.npz_dataset.NumpyDictDataset
:canonical: src.data.datasets.simulation.npz_dataset.NumpyDictDataset.load
:staticmethod:

```{autodoc2-docstring} src.data.datasets.simulation.npz_dataset.NumpyDictDataset.load
```

````

````{py:method} save(path: str) -> None
:canonical: src.data.datasets.simulation.npz_dataset.NumpyDictDataset.save

```{autodoc2-docstring} src.data.datasets.simulation.npz_dataset.NumpyDictDataset.save
```

````

`````
