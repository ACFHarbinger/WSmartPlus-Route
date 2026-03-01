# {py:mod}`src.data.datasets.simulation.gen_dataset`

```{py:module} src.data.datasets.simulation.gen_dataset
```

```{autodoc2-docstring} src.data.datasets.simulation.gen_dataset
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GenerativeDataset <src.data.datasets.simulation.gen_dataset.GenerativeDataset>`
  - ```{autodoc2-docstring} src.data.datasets.simulation.gen_dataset.GenerativeDataset
    :summary:
    ```
````

### API

`````{py:class} GenerativeDataset(data_dir: str, n_samples: int, n_days: int, n_bins: int, distribution: str = 'gamma1', depot: typing.Optional[numpy.ndarray] = None, locs: typing.Optional[numpy.ndarray] = None, noise_mean: float = 0.0, noise_variance: float = 0.0, max_waste: float = MAX_CAPACITY_PERCENT, grid: typing.Optional[typing.Any] = None)
:canonical: src.data.datasets.simulation.gen_dataset.GenerativeDataset

Bases: {py:obj}`src.data.datasets.simulation.sim_dataset.SimulationDataset`

```{autodoc2-docstring} src.data.datasets.simulation.gen_dataset.GenerativeDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.datasets.simulation.gen_dataset.GenerativeDataset.__init__
```

````{py:method} __len__() -> int
:canonical: src.data.datasets.simulation.gen_dataset.GenerativeDataset.__len__

```{autodoc2-docstring} src.data.datasets.simulation.gen_dataset.GenerativeDataset.__len__
```

````

````{py:method} __getitem__(index: int) -> typing.Dict[str, numpy.ndarray]
:canonical: src.data.datasets.simulation.gen_dataset.GenerativeDataset.__getitem__

```{autodoc2-docstring} src.data.datasets.simulation.gen_dataset.GenerativeDataset.__getitem__
```

````

````{py:method} _generate_waste_days() -> numpy.ndarray
:canonical: src.data.datasets.simulation.gen_dataset.GenerativeDataset._generate_waste_days

```{autodoc2-docstring} src.data.datasets.simulation.gen_dataset.GenerativeDataset._generate_waste_days
```

````

````{py:method} _apply_noise(waste: numpy.ndarray) -> numpy.ndarray
:canonical: src.data.datasets.simulation.gen_dataset.GenerativeDataset._apply_noise

```{autodoc2-docstring} src.data.datasets.simulation.gen_dataset.GenerativeDataset._apply_noise
```

````

`````
