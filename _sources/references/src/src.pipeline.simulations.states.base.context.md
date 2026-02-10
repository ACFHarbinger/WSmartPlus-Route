# {py:mod}`src.pipeline.simulations.states.base.context`

```{py:module} src.pipeline.simulations.states.base.context
```

```{autodoc2-docstring} src.pipeline.simulations.states.base.context
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationContext <src.pipeline.simulations.states.base.context.SimulationContext>`
  - ```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext
    :summary:
    ```
````

### API

`````{py:class} SimulationContext(opts: typing.Dict[str, typing.Any], device: torch.device, indices: typing.List[int], sample_id: int, pol_id: int, model_weights_path: str, variables_dict: typing.Dict[str, typing.Any])
:canonical: src.pipeline.simulations.states.base.context.SimulationContext

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.__init__
```

````{py:attribute} lock
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.lock
:type: typing.Optional[threading.Lock]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.lock
```

````

````{py:attribute} counter
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.counter
:type: typing.Optional[typing.Any]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.counter
```

````

````{py:attribute} overall_progress
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.overall_progress
:type: typing.Optional[typing.Any]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.overall_progress
```

````

````{py:attribute} shared_metrics
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.shared_metrics
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.shared_metrics
```

````

````{py:attribute} pbar
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.pbar
:type: typing.Optional[typing.Any]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.pbar
```

````

````{py:attribute} log_path
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.log_path
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.log_path
```

````

````{py:attribute} exec_time
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.exec_time
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.exec_time
```

````

````{py:attribute} start_time
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.start_time
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.start_time
```

````

````{py:attribute} end_time
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.end_time
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.end_time
```

````

````{py:attribute} pol_name
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.pol_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.pol_name
```

````

````{py:attribute} pol_engine
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.pol_engine
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.pol_engine
```

````

````{py:attribute} pol_threshold
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.pol_threshold
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.pol_threshold
```

````

````{py:attribute} pol_id
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.pol_id
:type: int
:value: >
   0

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.pol_id
```

````

````{py:attribute} pol_strip
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.pol_strip
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.pol_strip
```

````

````{py:method} _parse_policy_string() -> None
:canonical: src.pipeline.simulations.states.base.context.SimulationContext._parse_policy_string

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext._parse_policy_string
```

````

````{py:method} _extract_threshold(policy_key: str) -> None
:canonical: src.pipeline.simulations.states.base.context.SimulationContext._extract_threshold

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext._extract_threshold
```

````

````{py:method} _extract_threshold_with_config_char(policy_key: str, config_chars: typing.List[str]) -> None
:canonical: src.pipeline.simulations.states.base.context.SimulationContext._extract_threshold_with_config_char

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext._extract_threshold_with_config_char
```

````

````{py:method} _continue_init(variables_dict: typing.Dict[str, typing.Any], pol_id: int) -> None
:canonical: src.pipeline.simulations.states.base.context.SimulationContext._continue_init

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext._continue_init
```

````

````{py:method} transition_to(state: typing.Optional[src.pipeline.simulations.states.base.base.SimState]) -> None
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.transition_to

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.transition_to
```

````

````{py:method} run() -> typing.Optional[typing.Dict[str, typing.Any]]
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.run

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.run
```

````

````{py:method} get_current_state_tuple() -> typing.Tuple[typing.Any, ...]
:canonical: src.pipeline.simulations.states.base.context.SimulationContext.get_current_state_tuple

```{autodoc2-docstring} src.pipeline.simulations.states.base.context.SimulationContext.get_current_state_tuple
```

````

`````
