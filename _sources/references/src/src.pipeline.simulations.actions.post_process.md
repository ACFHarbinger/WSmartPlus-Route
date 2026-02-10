# {py:mod}`src.pipeline.simulations.actions.post_process`

```{py:module} src.pipeline.simulations.actions.post_process
```

```{autodoc2-docstring} src.pipeline.simulations.actions.post_process
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PostProcessAction <src.pipeline.simulations.actions.post_process.PostProcessAction>`
  - ```{autodoc2-docstring} src.pipeline.simulations.actions.post_process.PostProcessAction
    :summary:
    ```
````

### API

`````{py:class} PostProcessAction
:canonical: src.pipeline.simulations.actions.post_process.PostProcessAction

Bases: {py:obj}`src.pipeline.simulations.actions.base.SimulationAction`

```{autodoc2-docstring} src.pipeline.simulations.actions.post_process.PostProcessAction
```

````{py:method} execute(context: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.simulations.actions.post_process.PostProcessAction.execute

```{autodoc2-docstring} src.pipeline.simulations.actions.post_process.PostProcessAction.execute
```

````

````{py:method} _get_post_processing_configs(context: typing.Dict[str, typing.Any]) -> list
:canonical: src.pipeline.simulations.actions.post_process.PostProcessAction._get_post_processing_configs

```{autodoc2-docstring} src.pipeline.simulations.actions.post_process.PostProcessAction._get_post_processing_configs
```

````

````{py:method} _create_processors(entry: typing.Any, context: typing.Dict[str, typing.Any]) -> list
:canonical: src.pipeline.simulations.actions.post_process.PostProcessAction._create_processors

```{autodoc2-docstring} src.pipeline.simulations.actions.post_process.PostProcessAction._create_processors
```

````

````{py:method} _create_legacy_processors(item: typing.Any, context: typing.Dict[str, typing.Any], factory) -> list
:canonical: src.pipeline.simulations.actions.post_process.PostProcessAction._create_legacy_processors

```{autodoc2-docstring} src.pipeline.simulations.actions.post_process.PostProcessAction._create_legacy_processors
```

````

````{py:method} _apply_processors(processors: list, context: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.simulations.actions.post_process.PostProcessAction._apply_processors

```{autodoc2-docstring} src.pipeline.simulations.actions.post_process.PostProcessAction._apply_processors
```

````

`````
