# {py:mod}`src.data.fast_datasets`

```{py:module} src.data.fast_datasets
```

```{autodoc2-docstring} src.data.fast_datasets
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FastTdDataset <src.data.fast_datasets.FastTdDataset>`
  - ```{autodoc2-docstring} src.data.fast_datasets.FastTdDataset
    :summary:
    ```
* - {py:obj}`TensorDictDatasetFastGeneration <src.data.fast_datasets.TensorDictDatasetFastGeneration>`
  - ```{autodoc2-docstring} src.data.fast_datasets.TensorDictDatasetFastGeneration
    :summary:
    ```
````

### API

`````{py:class} FastTdDataset(td: tensordict.tensordict.TensorDict)
:canonical: src.data.fast_datasets.FastTdDataset

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} src.data.fast_datasets.FastTdDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.fast_datasets.FastTdDataset.__init__
```

````{py:method} __len__()
:canonical: src.data.fast_datasets.FastTdDataset.__len__

```{autodoc2-docstring} src.data.fast_datasets.FastTdDataset.__len__
```

````

````{py:method} __getitems__(idx)
:canonical: src.data.fast_datasets.FastTdDataset.__getitems__

```{autodoc2-docstring} src.data.fast_datasets.FastTdDataset.__getitems__
```

````

````{py:method} collate_fn(batch: typing.Union[dict, tensordict.tensordict.TensorDict])
:canonical: src.data.fast_datasets.FastTdDataset.collate_fn
:staticmethod:

```{autodoc2-docstring} src.data.fast_datasets.FastTdDataset.collate_fn
```

````

`````

`````{py:class} TensorDictDatasetFastGeneration(td: tensordict.tensordict.TensorDict)
:canonical: src.data.fast_datasets.TensorDictDatasetFastGeneration

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} src.data.fast_datasets.TensorDictDatasetFastGeneration
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.fast_datasets.TensorDictDatasetFastGeneration.__init__
```

````{py:method} __len__()
:canonical: src.data.fast_datasets.TensorDictDatasetFastGeneration.__len__

```{autodoc2-docstring} src.data.fast_datasets.TensorDictDatasetFastGeneration.__len__
```

````

````{py:method} __getitems__(index)
:canonical: src.data.fast_datasets.TensorDictDatasetFastGeneration.__getitems__

```{autodoc2-docstring} src.data.fast_datasets.TensorDictDatasetFastGeneration.__getitems__
```

````

````{py:method} collate_fn(batch: typing.Union[dict, tensordict.tensordict.TensorDict])
:canonical: src.data.fast_datasets.TensorDictDatasetFastGeneration.collate_fn
:staticmethod:

```{autodoc2-docstring} src.data.fast_datasets.TensorDictDatasetFastGeneration.collate_fn
```

````

`````
