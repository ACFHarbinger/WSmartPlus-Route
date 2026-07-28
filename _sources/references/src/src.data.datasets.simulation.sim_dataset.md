# {py:mod}`src.data.datasets.simulation.sim_dataset`

```{py:module} src.data.datasets.simulation.sim_dataset
```

```{autodoc2-docstring} src.data.datasets.simulation.sim_dataset
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationDataset <src.data.datasets.simulation.sim_dataset.SimulationDataset>`
  - ```{autodoc2-docstring} src.data.datasets.simulation.sim_dataset.SimulationDataset
    :summary:
    ```
````

### API

`````{py:class} SimulationDataset
:canonical: src.data.datasets.simulation.sim_dataset.SimulationDataset

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.data.datasets.simulation.sim_dataset.SimulationDataset
```

````{py:method} __len__() -> int
:canonical: src.data.datasets.simulation.sim_dataset.SimulationDataset.__len__
:abstractmethod:

```{autodoc2-docstring} src.data.datasets.simulation.sim_dataset.SimulationDataset.__len__
```

````

````{py:method} __getitem__(index: int) -> typing.Dict[str, numpy.ndarray]
:canonical: src.data.datasets.simulation.sim_dataset.SimulationDataset.__getitem__
:abstractmethod:

```{autodoc2-docstring} src.data.datasets.simulation.sim_dataset.SimulationDataset.__getitem__
```

````

`````
