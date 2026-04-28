# {py:mod}`src.configs.tasks.sim_hpo`

```{py:module} src.configs.tasks.sim_hpo
```

```{autodoc2-docstring} src.configs.tasks.sim_hpo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimHPOConfig <src.configs.tasks.sim_hpo.SimHPOConfig>`
  - ```{autodoc2-docstring} src.configs.tasks.sim_hpo.SimHPOConfig
    :summary:
    ```
````

### API

`````{py:class} SimHPOConfig
:canonical: src.configs.tasks.sim_hpo.SimHPOConfig

```{autodoc2-docstring} src.configs.tasks.sim_hpo.SimHPOConfig
```

````{py:attribute} method
:canonical: src.configs.tasks.sim_hpo.SimHPOConfig.method
:type: str
:value: >
   'tpe'

```{autodoc2-docstring} src.configs.tasks.sim_hpo.SimHPOConfig.method
```

````

````{py:attribute} metric
:canonical: src.configs.tasks.sim_hpo.SimHPOConfig.metric
:type: str
:value: >
   'profit'

```{autodoc2-docstring} src.configs.tasks.sim_hpo.SimHPOConfig.metric
```

````

````{py:attribute} metrics
:canonical: src.configs.tasks.sim_hpo.SimHPOConfig.metrics
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.sim_hpo.SimHPOConfig.metrics
```

````

````{py:attribute} n_trials
:canonical: src.configs.tasks.sim_hpo.SimHPOConfig.n_trials
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.tasks.sim_hpo.SimHPOConfig.n_trials
```

````

````{py:attribute} num_workers
:canonical: src.configs.tasks.sim_hpo.SimHPOConfig.num_workers
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.tasks.sim_hpo.SimHPOConfig.num_workers
```

````

````{py:attribute} search_space
:canonical: src.configs.tasks.sim_hpo.SimHPOConfig.search_space
:type: typing.Dict[str, typing.Dict[str, typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.sim_hpo.SimHPOConfig.search_space
```

````

````{py:attribute} policy_name
:canonical: src.configs.tasks.sim_hpo.SimHPOConfig.policy_name
:type: str
:value: >
   'alns'

```{autodoc2-docstring} src.configs.tasks.sim_hpo.SimHPOConfig.policy_name
```

````

````{py:attribute} selection_name
:canonical: src.configs.tasks.sim_hpo.SimHPOConfig.selection_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.sim_hpo.SimHPOConfig.selection_name
```

````

````{py:attribute} acceptance_name
:canonical: src.configs.tasks.sim_hpo.SimHPOConfig.acceptance_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.sim_hpo.SimHPOConfig.acceptance_name
```

````

````{py:attribute} improver_name
:canonical: src.configs.tasks.sim_hpo.SimHPOConfig.improver_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.sim_hpo.SimHPOConfig.improver_name
```

````

````{py:attribute} graph
:canonical: src.configs.tasks.sim_hpo.SimHPOConfig.graph
:type: logic.src.configs.envs.graph.GraphConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.sim_hpo.SimHPOConfig.graph
```

````

`````
