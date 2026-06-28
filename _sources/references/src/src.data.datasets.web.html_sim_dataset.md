# {py:mod}`src.data.datasets.web.html_sim_dataset`

```{py:module} src.data.datasets.web.html_sim_dataset
```

```{autodoc2-docstring} src.data.datasets.web.html_sim_dataset
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HtmlSimulationDataset <src.data.datasets.web.html_sim_dataset.HtmlSimulationDataset>`
  - ```{autodoc2-docstring} src.data.datasets.web.html_sim_dataset.HtmlSimulationDataset
    :summary:
    ```
````

### API

`````{py:class} HtmlSimulationDataset(sample: typing.Dict[str, typing.Any])
:canonical: src.data.datasets.web.html_sim_dataset.HtmlSimulationDataset

Bases: {py:obj}`logic.src.data.datasets.simulation.sim_dataset.SimulationDataset`

```{autodoc2-docstring} src.data.datasets.web.html_sim_dataset.HtmlSimulationDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.datasets.web.html_sim_dataset.HtmlSimulationDataset.__init__
```

````{py:method} __len__() -> int
:canonical: src.data.datasets.web.html_sim_dataset.HtmlSimulationDataset.__len__

```{autodoc2-docstring} src.data.datasets.web.html_sim_dataset.HtmlSimulationDataset.__len__
```

````

````{py:method} __getitem__(index: int) -> typing.Dict[str, typing.Any]
:canonical: src.data.datasets.web.html_sim_dataset.HtmlSimulationDataset.__getitem__

```{autodoc2-docstring} src.data.datasets.web.html_sim_dataset.HtmlSimulationDataset.__getitem__
```

````

````{py:method} load(path: str, area: typing.Optional[str] = None, waste_type: typing.Optional[str] = None, n_days: int = 31, n_bins: typing.Optional[int] = None) -> src.data.datasets.web.html_sim_dataset.HtmlSimulationDataset
:canonical: src.data.datasets.web.html_sim_dataset.HtmlSimulationDataset.load
:classmethod:

```{autodoc2-docstring} src.data.datasets.web.html_sim_dataset.HtmlSimulationDataset.load
```

````

`````
