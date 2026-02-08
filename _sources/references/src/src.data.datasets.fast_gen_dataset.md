# {py:mod}`src.data.datasets.fast_gen_dataset`

```{py:module} src.data.datasets.fast_gen_dataset
```

```{autodoc2-docstring} src.data.datasets.fast_gen_dataset
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TensorDictDatasetFastGeneration <src.data.datasets.fast_gen_dataset.TensorDictDatasetFastGeneration>`
  - ```{autodoc2-docstring} src.data.datasets.fast_gen_dataset.TensorDictDatasetFastGeneration
    :summary:
    ```
````

### API

`````{py:class} TensorDictDatasetFastGeneration(td: tensordict.tensordict.TensorDict)
:canonical: src.data.datasets.fast_gen_dataset.TensorDictDatasetFastGeneration

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} src.data.datasets.fast_gen_dataset.TensorDictDatasetFastGeneration
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.datasets.fast_gen_dataset.TensorDictDatasetFastGeneration.__init__
```

````{py:method} __len__()
:canonical: src.data.datasets.fast_gen_dataset.TensorDictDatasetFastGeneration.__len__

```{autodoc2-docstring} src.data.datasets.fast_gen_dataset.TensorDictDatasetFastGeneration.__len__
```

````

````{py:method} __getitems__(index)
:canonical: src.data.datasets.fast_gen_dataset.TensorDictDatasetFastGeneration.__getitems__

```{autodoc2-docstring} src.data.datasets.fast_gen_dataset.TensorDictDatasetFastGeneration.__getitems__
```

````

````{py:method} collate_fn(batch: typing.Any) -> typing.Any
:canonical: src.data.datasets.fast_gen_dataset.TensorDictDatasetFastGeneration.collate_fn
:staticmethod:

```{autodoc2-docstring} src.data.datasets.fast_gen_dataset.TensorDictDatasetFastGeneration.collate_fn
```

````

`````
