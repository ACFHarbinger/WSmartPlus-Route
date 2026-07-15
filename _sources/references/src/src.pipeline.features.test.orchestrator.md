# {py:mod}`src.pipeline.features.test.orchestrator`

```{py:module} src.pipeline.features.test.orchestrator
```

```{autodoc2-docstring} src.pipeline.features.test.orchestrator
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.pipeline.features.test.orchestrator.results_handler
src.pipeline.features.test.orchestrator.monitor
src.pipeline.features.test.orchestrator.parallel_runner
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_expand_other_ref <src.pipeline.features.test.orchestrator._expand_other_ref>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._expand_other_ref
    :summary:
    ```
* - {py:obj}`_expand_refs_recursive <src.pipeline.features.test.orchestrator._expand_refs_recursive>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._expand_refs_recursive
    :summary:
    ```
* - {py:obj}`_generate_pruned_config <src.pipeline.features.test.orchestrator._generate_pruned_config>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator._generate_pruned_config
    :summary:
    ```
* - {py:obj}`simulator_testing <src.pipeline.features.test.orchestrator.simulator_testing>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator.simulator_testing
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.pipeline.features.test.orchestrator.__all__>`
  - ```{autodoc2-docstring} src.pipeline.features.test.orchestrator.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: src.pipeline.features.test.orchestrator.__all__
:value: >
   ['simulator_testing', 'LoggerWriter', 'setup_logger_redirection', 'runs_per_policy', 'output_stats',...

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.__all__
```

````

````{py:function} _expand_other_ref(ref_dict: dict, root_dir: str) -> dict
:canonical: src.pipeline.features.test.orchestrator._expand_other_ref

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._expand_other_ref
```
````

````{py:function} _expand_refs_recursive(obj: typing.Any, root_dir: str) -> typing.Any
:canonical: src.pipeline.features.test.orchestrator._expand_refs_recursive

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._expand_refs_recursive
```
````

````{py:function} _generate_pruned_config(cfg: logic.src.configs.Config, root_dir: str) -> str
:canonical: src.pipeline.features.test.orchestrator._generate_pruned_config

```{autodoc2-docstring} src.pipeline.features.test.orchestrator._generate_pruned_config
```
````

````{py:function} simulator_testing(cfg: logic.src.configs.Config, data_size: int, device: typing.Any) -> None
:canonical: src.pipeline.features.test.orchestrator.simulator_testing

```{autodoc2-docstring} src.pipeline.features.test.orchestrator.simulator_testing
```
````
