# {py:mod}`src.configs.tasks.eval`

```{py:module} src.configs.tasks.eval
```

```{autodoc2-docstring} src.configs.tasks.eval
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EvalConfig <src.configs.tasks.eval.EvalConfig>`
  - ```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig
    :summary:
    ```
````

### API

`````{py:class} EvalConfig
:canonical: src.configs.tasks.eval.EvalConfig

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig
```

````{py:attribute} datasets
:canonical: src.configs.tasks.eval.EvalConfig.datasets
:type: typing.Optional[typing.List[str]]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.datasets
```

````

````{py:attribute} overwrite
:canonical: src.configs.tasks.eval.EvalConfig.overwrite
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.overwrite
```

````

````{py:attribute} output_filename
:canonical: src.configs.tasks.eval.EvalConfig.output_filename
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.output_filename
```

````

````{py:attribute} val_size
:canonical: src.configs.tasks.eval.EvalConfig.val_size
:type: int
:value: >
   12800

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.val_size
```

````

````{py:attribute} offset
:canonical: src.configs.tasks.eval.EvalConfig.offset
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.offset
```

````

````{py:attribute} eval_batch_size
:canonical: src.configs.tasks.eval.EvalConfig.eval_batch_size
:type: int
:value: >
   256

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.eval_batch_size
```

````

````{py:attribute} decoding
:canonical: src.configs.tasks.eval.EvalConfig.decoding
:type: src.configs.models.decoding.DecodingConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.decoding
```

````

````{py:attribute} policy
:canonical: src.configs.tasks.eval.EvalConfig.policy
:type: src.configs.policies.neural.NeuralConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.policy
```

````

````{py:attribute} seed
:canonical: src.configs.tasks.eval.EvalConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.seed
```

````

````{py:attribute} data_distribution
:canonical: src.configs.tasks.eval.EvalConfig.data_distribution
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.data_distribution
```

````

````{py:attribute} no_cuda
:canonical: src.configs.tasks.eval.EvalConfig.no_cuda
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.no_cuda
```

````

````{py:attribute} no_progress_bar
:canonical: src.configs.tasks.eval.EvalConfig.no_progress_bar
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.no_progress_bar
```

````

````{py:attribute} compress_mask
:canonical: src.configs.tasks.eval.EvalConfig.compress_mask
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.compress_mask
```

````

````{py:attribute} max_calc_batch_size
:canonical: src.configs.tasks.eval.EvalConfig.max_calc_batch_size
:type: int
:value: >
   12800

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.max_calc_batch_size
```

````

````{py:attribute} results_dir
:canonical: src.configs.tasks.eval.EvalConfig.results_dir
:type: str
:value: >
   'results'

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.results_dir
```

````

````{py:attribute} multiprocessing
:canonical: src.configs.tasks.eval.EvalConfig.multiprocessing
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.multiprocessing
```

````

````{py:attribute} graph
:canonical: src.configs.tasks.eval.EvalConfig.graph
:type: src.configs.envs.graph.GraphConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.graph
```

````

````{py:attribute} reward
:canonical: src.configs.tasks.eval.EvalConfig.reward
:type: src.configs.envs.objective.ObjectiveConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.reward
```

````

````{py:attribute} problem
:canonical: src.configs.tasks.eval.EvalConfig.problem
:type: str
:value: >
   'cwcvrp'

```{autodoc2-docstring} src.configs.tasks.eval.EvalConfig.problem
```

````

`````
