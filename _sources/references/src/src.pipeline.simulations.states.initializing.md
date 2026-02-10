# {py:mod}`src.pipeline.simulations.states.initializing`

```{py:module} src.pipeline.simulations.states.initializing
```

```{autodoc2-docstring} src.pipeline.simulations.states.initializing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`InitializingState <src.pipeline.simulations.states.initializing.InitializingState>`
  - ```{autodoc2-docstring} src.pipeline.simulations.states.initializing.InitializingState
    :summary:
    ```
````

### API

`````{py:class} InitializingState
:canonical: src.pipeline.simulations.states.initializing.InitializingState

Bases: {py:obj}`src.pipeline.simulations.states.base.SimState`

```{autodoc2-docstring} src.pipeline.simulations.states.initializing.InitializingState
```

````{py:method} handle(ctx: src.pipeline.simulations.states.base.SimulationContext) -> None
:canonical: src.pipeline.simulations.states.initializing.InitializingState.handle

```{autodoc2-docstring} src.pipeline.simulations.states.initializing.InitializingState.handle
```

````

````{py:method} _setup_logging_and_dirs(ctx)
:canonical: src.pipeline.simulations.states.initializing.InitializingState._setup_logging_and_dirs

```{autodoc2-docstring} src.pipeline.simulations.states.initializing.InitializingState._setup_logging_and_dirs
```

````

````{py:method} _load_all_configurations(ctx)
:canonical: src.pipeline.simulations.states.initializing.InitializingState._load_all_configurations

```{autodoc2-docstring} src.pipeline.simulations.states.initializing.InitializingState._load_all_configurations
```

````

````{py:method} _load_neural_configs(ctx)
:canonical: src.pipeline.simulations.states.initializing.InitializingState._load_neural_configs

```{autodoc2-docstring} src.pipeline.simulations.states.initializing.InitializingState._load_neural_configs
```

````

````{py:method} _setup_capacities(ctx)
:canonical: src.pipeline.simulations.states.initializing.InitializingState._setup_capacities

```{autodoc2-docstring} src.pipeline.simulations.states.initializing.InitializingState._setup_capacities
```

````

````{py:method} _load_checkpoint_if_needed(ctx)
:canonical: src.pipeline.simulations.states.initializing.InitializingState._load_checkpoint_if_needed

```{autodoc2-docstring} src.pipeline.simulations.states.initializing.InitializingState._load_checkpoint_if_needed
```

````

````{py:method} _setup_models(ctx)
:canonical: src.pipeline.simulations.states.initializing.InitializingState._setup_models

```{autodoc2-docstring} src.pipeline.simulations.states.initializing.InitializingState._setup_models
```

````

````{py:method} _restore_state(ctx, saved_state, last_day)
:canonical: src.pipeline.simulations.states.initializing.InitializingState._restore_state

```{autodoc2-docstring} src.pipeline.simulations.states.initializing.InitializingState._restore_state
```

````

````{py:method} _initialize_new_state(ctx, data, bins_coordinates, depot)
:canonical: src.pipeline.simulations.states.initializing.InitializingState._initialize_new_state

```{autodoc2-docstring} src.pipeline.simulations.states.initializing.InitializingState._initialize_new_state
```

````

````{py:method} _initialize_bins(ctx)
:canonical: src.pipeline.simulations.states.initializing.InitializingState._initialize_bins

```{autodoc2-docstring} src.pipeline.simulations.states.initializing.InitializingState._initialize_bins
```

````

`````
