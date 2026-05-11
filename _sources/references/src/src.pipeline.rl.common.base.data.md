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

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_cfg_get <src.pipeline.rl.common.base.data._cfg_get>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.base.data._cfg_get
    :summary:
    ```
* - {py:obj}`_get_eval_graphs <src.pipeline.rl.common.base.data._get_eval_graphs>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.base.data._get_eval_graphs
    :summary:
    ```
* - {py:obj}`_create_eval_env_and_gen <src.pipeline.rl.common.base.data._create_eval_env_and_gen>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.base.data._create_eval_env_and_gen
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

````{py:function} _cfg_get(obj: typing.Any, key: str, default: typing.Any = None) -> typing.Any
:canonical: src.pipeline.rl.common.base.data._cfg_get

```{autodoc2-docstring} src.pipeline.rl.common.base.data._cfg_get
```
````

````{py:function} _get_eval_graphs(cfg: typing.Any) -> list
:canonical: src.pipeline.rl.common.base.data._get_eval_graphs

```{autodoc2-docstring} src.pipeline.rl.common.base.data._get_eval_graphs
```
````

````{py:function} _create_eval_env_and_gen(cfg: typing.Any, eval_graph: typing.Any) -> tuple
:canonical: src.pipeline.rl.common.base.data._create_eval_env_and_gen

```{autodoc2-docstring} src.pipeline.rl.common.base.data._create_eval_env_and_gen
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

````{py:method} val_dataloader() -> typing.Union[torch.utils.data.DataLoader, typing.List[torch.utils.data.DataLoader]]
:canonical: src.pipeline.rl.common.base.data.DataMixin.val_dataloader

```{autodoc2-docstring} src.pipeline.rl.common.base.data.DataMixin.val_dataloader
```

````

`````
