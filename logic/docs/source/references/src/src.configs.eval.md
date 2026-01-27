# {py:mod}`src.configs.eval`

```{py:module} src.configs.eval
```

```{autodoc2-docstring} src.configs.eval
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EvalConfig <src.configs.eval.EvalConfig>`
  - ```{autodoc2-docstring} src.configs.eval.EvalConfig
    :summary:
    ```
````

### API

`````{py:class} EvalConfig
:canonical: src.configs.eval.EvalConfig

```{autodoc2-docstring} src.configs.eval.EvalConfig
```

````{py:attribute} datasets
:canonical: src.configs.eval.EvalConfig.datasets
:type: typing.Optional[typing.List[str]]
:value: >
   None

```{autodoc2-docstring} src.configs.eval.EvalConfig.datasets
```

````

````{py:attribute} overwrite
:canonical: src.configs.eval.EvalConfig.overwrite
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.eval.EvalConfig.overwrite
```

````

````{py:attribute} output_filename
:canonical: src.configs.eval.EvalConfig.output_filename
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.eval.EvalConfig.output_filename
```

````

````{py:attribute} val_size
:canonical: src.configs.eval.EvalConfig.val_size
:type: int
:value: >
   12800

```{autodoc2-docstring} src.configs.eval.EvalConfig.val_size
```

````

````{py:attribute} offset
:canonical: src.configs.eval.EvalConfig.offset
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.eval.EvalConfig.offset
```

````

````{py:attribute} eval_batch_size
:canonical: src.configs.eval.EvalConfig.eval_batch_size
:type: int
:value: >
   256

```{autodoc2-docstring} src.configs.eval.EvalConfig.eval_batch_size
```

````

````{py:attribute} decode_type
:canonical: src.configs.eval.EvalConfig.decode_type
:type: str
:value: >
   'greedy'

```{autodoc2-docstring} src.configs.eval.EvalConfig.decode_type
```

````

````{py:attribute} width
:canonical: src.configs.eval.EvalConfig.width
:type: typing.Optional[typing.List[int]]
:value: >
   None

```{autodoc2-docstring} src.configs.eval.EvalConfig.width
```

````

````{py:attribute} decode_strategy
:canonical: src.configs.eval.EvalConfig.decode_strategy
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.eval.EvalConfig.decode_strategy
```

````

````{py:attribute} softmax_temperature
:canonical: src.configs.eval.EvalConfig.softmax_temperature
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.eval.EvalConfig.softmax_temperature
```

````

````{py:attribute} model
:canonical: src.configs.eval.EvalConfig.model
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.eval.EvalConfig.model
```

````

````{py:attribute} seed
:canonical: src.configs.eval.EvalConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.eval.EvalConfig.seed
```

````

````{py:attribute} data_distribution
:canonical: src.configs.eval.EvalConfig.data_distribution
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.eval.EvalConfig.data_distribution
```

````

````{py:attribute} no_cuda
:canonical: src.configs.eval.EvalConfig.no_cuda
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.eval.EvalConfig.no_cuda
```

````

````{py:attribute} no_progress_bar
:canonical: src.configs.eval.EvalConfig.no_progress_bar
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.eval.EvalConfig.no_progress_bar
```

````

````{py:attribute} compress_mask
:canonical: src.configs.eval.EvalConfig.compress_mask
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.eval.EvalConfig.compress_mask
```

````

````{py:attribute} max_calc_batch_size
:canonical: src.configs.eval.EvalConfig.max_calc_batch_size
:type: int
:value: >
   12800

```{autodoc2-docstring} src.configs.eval.EvalConfig.max_calc_batch_size
```

````

````{py:attribute} results_dir
:canonical: src.configs.eval.EvalConfig.results_dir
:type: str
:value: >
   'results'

```{autodoc2-docstring} src.configs.eval.EvalConfig.results_dir
```

````

````{py:attribute} multiprocessing
:canonical: src.configs.eval.EvalConfig.multiprocessing
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.eval.EvalConfig.multiprocessing
```

````

````{py:attribute} graph_size
:canonical: src.configs.eval.EvalConfig.graph_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.eval.EvalConfig.graph_size
```

````

````{py:attribute} area
:canonical: src.configs.eval.EvalConfig.area
:type: str
:value: >
   'riomaior'

```{autodoc2-docstring} src.configs.eval.EvalConfig.area
```

````

````{py:attribute} waste_type
:canonical: src.configs.eval.EvalConfig.waste_type
:type: str
:value: >
   'plastic'

```{autodoc2-docstring} src.configs.eval.EvalConfig.waste_type
```

````

````{py:attribute} focus_graph
:canonical: src.configs.eval.EvalConfig.focus_graph
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.eval.EvalConfig.focus_graph
```

````

````{py:attribute} focus_size
:canonical: src.configs.eval.EvalConfig.focus_size
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.eval.EvalConfig.focus_size
```

````

````{py:attribute} edge_threshold
:canonical: src.configs.eval.EvalConfig.edge_threshold
:type: str
:value: >
   '0'

```{autodoc2-docstring} src.configs.eval.EvalConfig.edge_threshold
```

````

````{py:attribute} edge_method
:canonical: src.configs.eval.EvalConfig.edge_method
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.eval.EvalConfig.edge_method
```

````

````{py:attribute} distance_method
:canonical: src.configs.eval.EvalConfig.distance_method
:type: str
:value: >
   'ogd'

```{autodoc2-docstring} src.configs.eval.EvalConfig.distance_method
```

````

````{py:attribute} dm_filepath
:canonical: src.configs.eval.EvalConfig.dm_filepath
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.eval.EvalConfig.dm_filepath
```

````

````{py:attribute} vertex_method
:canonical: src.configs.eval.EvalConfig.vertex_method
:type: str
:value: >
   'mmn'

```{autodoc2-docstring} src.configs.eval.EvalConfig.vertex_method
```

````

````{py:attribute} w_length
:canonical: src.configs.eval.EvalConfig.w_length
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.eval.EvalConfig.w_length
```

````

````{py:attribute} w_waste
:canonical: src.configs.eval.EvalConfig.w_waste
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.eval.EvalConfig.w_waste
```

````

````{py:attribute} w_overflows
:canonical: src.configs.eval.EvalConfig.w_overflows
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.eval.EvalConfig.w_overflows
```

````

````{py:attribute} problem
:canonical: src.configs.eval.EvalConfig.problem
:type: str
:value: >
   'cwcvrp'

```{autodoc2-docstring} src.configs.eval.EvalConfig.problem
```

````

````{py:attribute} encoder
:canonical: src.configs.eval.EvalConfig.encoder
:type: str
:value: >
   'gat'

```{autodoc2-docstring} src.configs.eval.EvalConfig.encoder
```

````

````{py:attribute} load_path
:canonical: src.configs.eval.EvalConfig.load_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.eval.EvalConfig.load_path
```

````

`````
