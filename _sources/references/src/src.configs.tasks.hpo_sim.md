# {py:mod}`src.configs.tasks.hpo_sim`

```{py:module} src.configs.tasks.hpo_sim
```

```{autodoc2-docstring} src.configs.tasks.hpo_sim
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimHPOConfig <src.configs.tasks.hpo_sim.SimHPOConfig>`
  - ```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig
    :summary:
    ```
````

### API

`````{py:class} SimHPOConfig
:canonical: src.configs.tasks.hpo_sim.SimHPOConfig

```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig
```

````{py:attribute} method
:canonical: src.configs.tasks.hpo_sim.SimHPOConfig.method
:type: str
:value: >
   'tpe'

```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig.method
```

````

````{py:attribute} metric
:canonical: src.configs.tasks.hpo_sim.SimHPOConfig.metric
:type: str
:value: >
   'profit'

```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig.metric
```

````

````{py:attribute} metrics
:canonical: src.configs.tasks.hpo_sim.SimHPOConfig.metrics
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig.metrics
```

````

````{py:attribute} n_trials
:canonical: src.configs.tasks.hpo_sim.SimHPOConfig.n_trials
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig.n_trials
```

````

````{py:attribute} num_workers
:canonical: src.configs.tasks.hpo_sim.SimHPOConfig.num_workers
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig.num_workers
```

````

````{py:attribute} search_space
:canonical: src.configs.tasks.hpo_sim.SimHPOConfig.search_space
:type: typing.Dict[str, typing.Dict[str, typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig.search_space
```

````

````{py:attribute} policy_name
:canonical: src.configs.tasks.hpo_sim.SimHPOConfig.policy_name
:type: str
:value: >
   'alns'

```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig.policy_name
```

````

````{py:attribute} selection_name
:canonical: src.configs.tasks.hpo_sim.SimHPOConfig.selection_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig.selection_name
```

````

````{py:attribute} acceptance_name
:canonical: src.configs.tasks.hpo_sim.SimHPOConfig.acceptance_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig.acceptance_name
```

````

````{py:attribute} improver_name
:canonical: src.configs.tasks.hpo_sim.SimHPOConfig.improver_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig.improver_name
```

````

````{py:attribute} policy_keywords
:canonical: src.configs.tasks.hpo_sim.SimHPOConfig.policy_keywords
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig.policy_keywords
```

````

````{py:attribute} selection_keywords
:canonical: src.configs.tasks.hpo_sim.SimHPOConfig.selection_keywords
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig.selection_keywords
```

````

````{py:attribute} acceptance_keywords
:canonical: src.configs.tasks.hpo_sim.SimHPOConfig.acceptance_keywords
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig.acceptance_keywords
```

````

````{py:attribute} improver_keywords
:canonical: src.configs.tasks.hpo_sim.SimHPOConfig.improver_keywords
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig.improver_keywords
```

````

````{py:attribute} graph
:canonical: src.configs.tasks.hpo_sim.SimHPOConfig.graph
:type: logic.src.configs.envs.graph.GraphConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.hpo_sim.SimHPOConfig.graph
```

````

`````
