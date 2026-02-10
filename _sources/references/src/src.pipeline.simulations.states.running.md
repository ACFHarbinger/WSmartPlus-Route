# {py:mod}`src.pipeline.simulations.states.running`

```{py:module} src.pipeline.simulations.states.running
```

```{autodoc2-docstring} src.pipeline.simulations.states.running
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RunningState <src.pipeline.simulations.states.running.RunningState>`
  - ```{autodoc2-docstring} src.pipeline.simulations.states.running.RunningState
    :summary:
    ```
````

### API

`````{py:class} RunningState
:canonical: src.pipeline.simulations.states.running.RunningState

Bases: {py:obj}`src.pipeline.simulations.states.base.SimState`

```{autodoc2-docstring} src.pipeline.simulations.states.running.RunningState
```

````{py:method} handle(ctx: src.pipeline.simulations.states.base.SimulationContext) -> None
:canonical: src.pipeline.simulations.states.running.RunningState.handle

```{autodoc2-docstring} src.pipeline.simulations.states.running.RunningState.handle
```

````

````{py:method} _run_simulation_days(ctx, iterator, hook, realtime_log_path)
:canonical: src.pipeline.simulations.states.running.RunningState._run_simulation_days

```{autodoc2-docstring} src.pipeline.simulations.states.running.RunningState._run_simulation_days
```

````

````{py:method} _get_current_policy_config(ctx)
:canonical: src.pipeline.simulations.states.running.RunningState._get_current_policy_config

```{autodoc2-docstring} src.pipeline.simulations.states.running.RunningState._get_current_policy_config
```

````

````{py:method} _create_day_context(ctx, day, current_policy_config, realtime_log_path)
:canonical: src.pipeline.simulations.states.running.RunningState._create_day_context

```{autodoc2-docstring} src.pipeline.simulations.states.running.RunningState._create_day_context
```

````

````{py:method} _update_ctx_from_day_context(ctx, day_context)
:canonical: src.pipeline.simulations.states.running.RunningState._update_ctx_from_day_context

```{autodoc2-docstring} src.pipeline.simulations.states.running.RunningState._update_ctx_from_day_context
```

````

````{py:method} _update_metrics(ctx, day, output_dict, dlog)
:canonical: src.pipeline.simulations.states.running.RunningState._update_metrics

```{autodoc2-docstring} src.pipeline.simulations.states.running.RunningState._update_metrics
```

````

`````
