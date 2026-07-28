# {py:mod}`src.configs.tasks.batch`

```{py:module} src.configs.tasks.batch
```

```{autodoc2-docstring} src.configs.tasks.batch
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BatchStepConfig <src.configs.tasks.batch.BatchStepConfig>`
  - ```{autodoc2-docstring} src.configs.tasks.batch.BatchStepConfig
    :summary:
    ```
* - {py:obj}`BatchRunConfig <src.configs.tasks.batch.BatchRunConfig>`
  - ```{autodoc2-docstring} src.configs.tasks.batch.BatchRunConfig
    :summary:
    ```
* - {py:obj}`BatchConfig <src.configs.tasks.batch.BatchConfig>`
  - ```{autodoc2-docstring} src.configs.tasks.batch.BatchConfig
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.configs.tasks.batch.__all__>`
  - ```{autodoc2-docstring} src.configs.tasks.batch.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: src.configs.tasks.batch.__all__
:value: >
   ['BatchConfig', 'BatchRunConfig', 'BatchStepConfig']

```{autodoc2-docstring} src.configs.tasks.batch.__all__
```

````

`````{py:class} BatchStepConfig
:canonical: src.configs.tasks.batch.BatchStepConfig

```{autodoc2-docstring} src.configs.tasks.batch.BatchStepConfig
```

````{py:attribute} type
:canonical: src.configs.tasks.batch.BatchStepConfig.type
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} src.configs.tasks.batch.BatchStepConfig.type
```

````

````{py:attribute} args
:canonical: src.configs.tasks.batch.BatchStepConfig.args
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.batch.BatchStepConfig.args
```

````

````{py:attribute} condition
:canonical: src.configs.tasks.batch.BatchStepConfig.condition
:type: str
:value: >
   'always'

```{autodoc2-docstring} src.configs.tasks.batch.BatchStepConfig.condition
```

````

````{py:attribute} name
:canonical: src.configs.tasks.batch.BatchStepConfig.name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.batch.BatchStepConfig.name
```

````

`````

`````{py:class} BatchRunConfig
:canonical: src.configs.tasks.batch.BatchRunConfig

```{autodoc2-docstring} src.configs.tasks.batch.BatchRunConfig
```

````{py:attribute} task
:canonical: src.configs.tasks.batch.BatchRunConfig.task
:type: str
:value: >
   'test_sim'

```{autodoc2-docstring} src.configs.tasks.batch.BatchRunConfig.task
```

````

````{py:attribute} name
:canonical: src.configs.tasks.batch.BatchRunConfig.name
:type: str
:value: >
   'run'

```{autodoc2-docstring} src.configs.tasks.batch.BatchRunConfig.name
```

````

````{py:attribute} overrides
:canonical: src.configs.tasks.batch.BatchRunConfig.overrides
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.batch.BatchRunConfig.overrides
```

````

````{py:attribute} pre_steps
:canonical: src.configs.tasks.batch.BatchRunConfig.pre_steps
:type: typing.List[src.configs.tasks.batch.BatchStepConfig]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.batch.BatchRunConfig.pre_steps
```

````

````{py:attribute} post_steps
:canonical: src.configs.tasks.batch.BatchRunConfig.post_steps
:type: typing.List[src.configs.tasks.batch.BatchStepConfig]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.batch.BatchRunConfig.post_steps
```

````

````{py:attribute} expand
:canonical: src.configs.tasks.batch.BatchRunConfig.expand
:type: typing.Dict[str, typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.batch.BatchRunConfig.expand
```

````

````{py:attribute} base_overrides
:canonical: src.configs.tasks.batch.BatchRunConfig.base_overrides
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.batch.BatchRunConfig.base_overrides
```

````

````{py:attribute} name_template
:canonical: src.configs.tasks.batch.BatchRunConfig.name_template
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.batch.BatchRunConfig.name_template
```

````

````{py:attribute} dim_overrides
:canonical: src.configs.tasks.batch.BatchRunConfig.dim_overrides
:type: typing.Dict[str, str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.batch.BatchRunConfig.dim_overrides
```

````

````{py:attribute} metadata
:canonical: src.configs.tasks.batch.BatchRunConfig.metadata
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.batch.BatchRunConfig.metadata
```

````

`````

`````{py:class} BatchConfig
:canonical: src.configs.tasks.batch.BatchConfig

```{autodoc2-docstring} src.configs.tasks.batch.BatchConfig
```

````{py:attribute} name
:canonical: src.configs.tasks.batch.BatchConfig.name
:type: str
:value: >
   'batch'

```{autodoc2-docstring} src.configs.tasks.batch.BatchConfig.name
```

````

````{py:attribute} fail_fast
:canonical: src.configs.tasks.batch.BatchConfig.fail_fast
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.batch.BatchConfig.fail_fast
```

````

````{py:attribute} dry_run
:canonical: src.configs.tasks.batch.BatchConfig.dry_run
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.batch.BatchConfig.dry_run
```

````

````{py:attribute} setup
:canonical: src.configs.tasks.batch.BatchConfig.setup
:type: typing.List[src.configs.tasks.batch.BatchStepConfig]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.batch.BatchConfig.setup
```

````

````{py:attribute} teardown
:canonical: src.configs.tasks.batch.BatchConfig.teardown
:type: typing.List[src.configs.tasks.batch.BatchStepConfig]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.batch.BatchConfig.teardown
```

````

````{py:attribute} runs
:canonical: src.configs.tasks.batch.BatchConfig.runs
:type: typing.List[src.configs.tasks.batch.BatchRunConfig]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.batch.BatchConfig.runs
```

````

`````
