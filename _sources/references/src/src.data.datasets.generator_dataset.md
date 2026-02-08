# {py:mod}`src.data.datasets.generator_dataset`

```{py:module} src.data.datasets.generator_dataset
```

```{autodoc2-docstring} src.data.datasets.generator_dataset
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GeneratorDataset <src.data.datasets.generator_dataset.GeneratorDataset>`
  - ```{autodoc2-docstring} src.data.datasets.generator_dataset.GeneratorDataset
    :summary:
    ```
````

### API

`````{py:class} GeneratorDataset(generator, size: int)
:canonical: src.data.datasets.generator_dataset.GeneratorDataset

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} src.data.datasets.generator_dataset.GeneratorDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.datasets.generator_dataset.GeneratorDataset.__init__
```

````{py:method} __len__() -> int
:canonical: src.data.datasets.generator_dataset.GeneratorDataset.__len__

```{autodoc2-docstring} src.data.datasets.generator_dataset.GeneratorDataset.__len__
```

````

````{py:method} __getitem__(index: int) -> tensordict.tensordict.TensorDict
:canonical: src.data.datasets.generator_dataset.GeneratorDataset.__getitem__

```{autodoc2-docstring} src.data.datasets.generator_dataset.GeneratorDataset.__getitem__
```

````

`````
