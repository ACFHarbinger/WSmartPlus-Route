# {py:mod}`src.data.datasets.td_dataset`

```{py:module} src.data.datasets.td_dataset
```

```{autodoc2-docstring} src.data.datasets.td_dataset
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TensorDictDataset <src.data.datasets.td_dataset.TensorDictDataset>`
  - ```{autodoc2-docstring} src.data.datasets.td_dataset.TensorDictDataset
    :summary:
    ```
````

### API

`````{py:class} TensorDictDataset(td: tensordict.tensordict.TensorDict)
:canonical: src.data.datasets.td_dataset.TensorDictDataset

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} src.data.datasets.td_dataset.TensorDictDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.datasets.td_dataset.TensorDictDataset.__init__
```

````{py:method} __len__()
:canonical: src.data.datasets.td_dataset.TensorDictDataset.__len__

```{autodoc2-docstring} src.data.datasets.td_dataset.TensorDictDataset.__len__
```

````

````{py:method} __getitem__(index)
:canonical: src.data.datasets.td_dataset.TensorDictDataset.__getitem__

```{autodoc2-docstring} src.data.datasets.td_dataset.TensorDictDataset.__getitem__
```

````

````{py:method} load(path: str)
:canonical: src.data.datasets.td_dataset.TensorDictDataset.load
:staticmethod:

```{autodoc2-docstring} src.data.datasets.td_dataset.TensorDictDataset.load
```

````

````{py:method} save(path: str)
:canonical: src.data.datasets.td_dataset.TensorDictDataset.save

```{autodoc2-docstring} src.data.datasets.td_dataset.TensorDictDataset.save
```

````

`````
