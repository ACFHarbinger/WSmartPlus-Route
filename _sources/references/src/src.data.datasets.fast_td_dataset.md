# {py:mod}`src.data.datasets.fast_td_dataset`

```{py:module} src.data.datasets.fast_td_dataset
```

```{autodoc2-docstring} src.data.datasets.fast_td_dataset
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FastTdDataset <src.data.datasets.fast_td_dataset.FastTdDataset>`
  - ```{autodoc2-docstring} src.data.datasets.fast_td_dataset.FastTdDataset
    :summary:
    ```
````

### API

`````{py:class} FastTdDataset(td: tensordict.tensordict.TensorDict)
:canonical: src.data.datasets.fast_td_dataset.FastTdDataset

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} src.data.datasets.fast_td_dataset.FastTdDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.datasets.fast_td_dataset.FastTdDataset.__init__
```

````{py:method} __len__()
:canonical: src.data.datasets.fast_td_dataset.FastTdDataset.__len__

```{autodoc2-docstring} src.data.datasets.fast_td_dataset.FastTdDataset.__len__
```

````

````{py:method} __getitems__(index)
:canonical: src.data.datasets.fast_td_dataset.FastTdDataset.__getitems__

```{autodoc2-docstring} src.data.datasets.fast_td_dataset.FastTdDataset.__getitems__
```

````

````{py:method} collate_fn(batch: typing.Any) -> typing.Any
:canonical: src.data.datasets.fast_td_dataset.FastTdDataset.collate_fn
:staticmethod:

```{autodoc2-docstring} src.data.datasets.fast_td_dataset.FastTdDataset.collate_fn
```

````

`````
