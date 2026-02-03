# {py:mod}`src.data.datasets`

```{py:module} src.data.datasets
```

```{autodoc2-docstring} src.data.datasets
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TensorDictDataset <src.data.datasets.TensorDictDataset>`
  - ```{autodoc2-docstring} src.data.datasets.TensorDictDataset
    :summary:
    ```
* - {py:obj}`FastTdDataset <src.data.datasets.FastTdDataset>`
  - ```{autodoc2-docstring} src.data.datasets.FastTdDataset
    :summary:
    ```
* - {py:obj}`TensorDictDatasetFastGeneration <src.data.datasets.TensorDictDatasetFastGeneration>`
  - ```{autodoc2-docstring} src.data.datasets.TensorDictDatasetFastGeneration
    :summary:
    ```
* - {py:obj}`GeneratorDataset <src.data.datasets.GeneratorDataset>`
  - ```{autodoc2-docstring} src.data.datasets.GeneratorDataset
    :summary:
    ```
* - {py:obj}`ExtraKeyDataset <src.data.datasets.ExtraKeyDataset>`
  - ```{autodoc2-docstring} src.data.datasets.ExtraKeyDataset
    :summary:
    ```
* - {py:obj}`BaselineDataset <src.data.datasets.BaselineDataset>`
  - ```{autodoc2-docstring} src.data.datasets.BaselineDataset
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`tensordict_collate_fn <src.data.datasets.tensordict_collate_fn>`
  - ```{autodoc2-docstring} src.data.datasets.tensordict_collate_fn
    :summary:
    ```
````

### API

`````{py:class} TensorDictDataset(td: tensordict.tensordict.TensorDict)
:canonical: src.data.datasets.TensorDictDataset

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} src.data.datasets.TensorDictDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.datasets.TensorDictDataset.__init__
```

````{py:method} __len__()
:canonical: src.data.datasets.TensorDictDataset.__len__

```{autodoc2-docstring} src.data.datasets.TensorDictDataset.__len__
```

````

````{py:method} __getitem__(index)
:canonical: src.data.datasets.TensorDictDataset.__getitem__

```{autodoc2-docstring} src.data.datasets.TensorDictDataset.__getitem__
```

````

````{py:method} load(path: str)
:canonical: src.data.datasets.TensorDictDataset.load
:staticmethod:

```{autodoc2-docstring} src.data.datasets.TensorDictDataset.load
```

````

````{py:method} save(path: str)
:canonical: src.data.datasets.TensorDictDataset.save

```{autodoc2-docstring} src.data.datasets.TensorDictDataset.save
```

````

`````

`````{py:class} FastTdDataset(td: tensordict.tensordict.TensorDict)
:canonical: src.data.datasets.FastTdDataset

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} src.data.datasets.FastTdDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.datasets.FastTdDataset.__init__
```

````{py:method} __len__()
:canonical: src.data.datasets.FastTdDataset.__len__

```{autodoc2-docstring} src.data.datasets.FastTdDataset.__len__
```

````

````{py:method} __getitems__(index)
:canonical: src.data.datasets.FastTdDataset.__getitems__

```{autodoc2-docstring} src.data.datasets.FastTdDataset.__getitems__
```

````

````{py:method} collate_fn(batch: typing.Any) -> typing.Any
:canonical: src.data.datasets.FastTdDataset.collate_fn
:staticmethod:

```{autodoc2-docstring} src.data.datasets.FastTdDataset.collate_fn
```

````

`````

`````{py:class} TensorDictDatasetFastGeneration(td: tensordict.tensordict.TensorDict)
:canonical: src.data.datasets.TensorDictDatasetFastGeneration

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} src.data.datasets.TensorDictDatasetFastGeneration
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.datasets.TensorDictDatasetFastGeneration.__init__
```

````{py:method} __len__()
:canonical: src.data.datasets.TensorDictDatasetFastGeneration.__len__

```{autodoc2-docstring} src.data.datasets.TensorDictDatasetFastGeneration.__len__
```

````

````{py:method} __getitems__(index)
:canonical: src.data.datasets.TensorDictDatasetFastGeneration.__getitems__

```{autodoc2-docstring} src.data.datasets.TensorDictDatasetFastGeneration.__getitems__
```

````

````{py:method} collate_fn(batch: typing.Any) -> typing.Any
:canonical: src.data.datasets.TensorDictDatasetFastGeneration.collate_fn
:staticmethod:

```{autodoc2-docstring} src.data.datasets.TensorDictDatasetFastGeneration.collate_fn
```

````

`````

`````{py:class} GeneratorDataset(generator, size: int)
:canonical: src.data.datasets.GeneratorDataset

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} src.data.datasets.GeneratorDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.datasets.GeneratorDataset.__init__
```

````{py:method} __len__() -> int
:canonical: src.data.datasets.GeneratorDataset.__len__

```{autodoc2-docstring} src.data.datasets.GeneratorDataset.__len__
```

````

````{py:method} __getitem__(index: int) -> tensordict.tensordict.TensorDict
:canonical: src.data.datasets.GeneratorDataset.__getitem__

```{autodoc2-docstring} src.data.datasets.GeneratorDataset.__getitem__
```

````

`````

`````{py:class} ExtraKeyDataset(dataset: torch.utils.data.Dataset, extra: dict[str, torch.Tensor])
:canonical: src.data.datasets.ExtraKeyDataset

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} src.data.datasets.ExtraKeyDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.datasets.ExtraKeyDataset.__init__
```

````{py:method} __getitem__(index: int) -> dict
:canonical: src.data.datasets.ExtraKeyDataset.__getitem__

```{autodoc2-docstring} src.data.datasets.ExtraKeyDataset.__getitem__
```

````

````{py:method} __len__() -> int
:canonical: src.data.datasets.ExtraKeyDataset.__len__

```{autodoc2-docstring} src.data.datasets.ExtraKeyDataset.__len__
```

````

`````

`````{py:class} BaselineDataset(dataset: torch.utils.data.Dataset, baseline: torch.Tensor)
:canonical: src.data.datasets.BaselineDataset

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} src.data.datasets.BaselineDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.datasets.BaselineDataset.__init__
```

````{py:method} __getitem__(index: int) -> dict
:canonical: src.data.datasets.BaselineDataset.__getitem__

```{autodoc2-docstring} src.data.datasets.BaselineDataset.__getitem__
```

````

````{py:method} __len__() -> int
:canonical: src.data.datasets.BaselineDataset.__len__

```{autodoc2-docstring} src.data.datasets.BaselineDataset.__len__
```

````

`````

````{py:function} tensordict_collate_fn(batch: list[typing.Union[dict, tensordict.tensordict.TensorDict]]) -> typing.Union[dict, tensordict.tensordict.TensorDict]
:canonical: src.data.datasets.tensordict_collate_fn

```{autodoc2-docstring} src.data.datasets.tensordict_collate_fn
```
````
