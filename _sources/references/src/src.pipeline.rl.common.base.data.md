# {py:mod}`src.pipeline.rl.common.base.data`

```{py:module} src.pipeline.rl.common.base.data
```

```{autodoc2-docstring} src.pipeline.rl.common.base.data
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DataMixin <src.pipeline.rl.common.base.data.DataMixin>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.base.data.DataMixin
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.rl.common.base.data.logger>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.base.data.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.rl.common.base.data.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.rl.common.base.data.logger
```

````

`````{py:class} DataMixin()
:canonical: src.pipeline.rl.common.base.data.DataMixin

```{autodoc2-docstring} src.pipeline.rl.common.base.data.DataMixin
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.base.data.DataMixin.__init__
```

````{py:method} setup(stage: str) -> None
:canonical: src.pipeline.rl.common.base.data.DataMixin.setup

```{autodoc2-docstring} src.pipeline.rl.common.base.data.DataMixin.setup
```

````

````{py:method} train_dataloader() -> torch.utils.data.DataLoader
:canonical: src.pipeline.rl.common.base.data.DataMixin.train_dataloader

```{autodoc2-docstring} src.pipeline.rl.common.base.data.DataMixin.train_dataloader
```

````

````{py:method} val_dataloader() -> torch.utils.data.DataLoader
:canonical: src.pipeline.rl.common.base.data.DataMixin.val_dataloader

```{autodoc2-docstring} src.pipeline.rl.common.base.data.DataMixin.val_dataloader
```

````

`````
