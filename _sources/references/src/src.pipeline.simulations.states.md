# {py:mod}`src.pipeline.simulations.states`

```{py:module} src.pipeline.simulations.states
```

```{autodoc2-docstring} src.pipeline.simulations.states
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationContext <src.pipeline.simulations.states.SimulationContext>`
  - ```{autodoc2-docstring} src.pipeline.simulations.states.SimulationContext
    :summary:
    ```
* - {py:obj}`SimState <src.pipeline.simulations.states.SimState>`
  - ```{autodoc2-docstring} src.pipeline.simulations.states.SimState
    :summary:
    ```
* - {py:obj}`InitializingState <src.pipeline.simulations.states.InitializingState>`
  - ```{autodoc2-docstring} src.pipeline.simulations.states.InitializingState
    :summary:
    ```
* - {py:obj}`RunningState <src.pipeline.simulations.states.RunningState>`
  - ```{autodoc2-docstring} src.pipeline.simulations.states.RunningState
    :summary:
    ```
* - {py:obj}`FinishingState <src.pipeline.simulations.states.FinishingState>`
  - ```{autodoc2-docstring} src.pipeline.simulations.states.FinishingState
    :summary:
    ```
````

### API

`````{py:class} SimulationContext(opts: typing.Dict[str, typing.Any], device: torch.device, indices: typing.List[int], sample_id: int, pol_id: int, model_weights_path: str, variables_dict: typing.Dict[str, typing.Any])
:canonical: src.pipeline.simulations.states.SimulationContext

```{autodoc2-docstring} src.pipeline.simulations.states.SimulationContext
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.states.SimulationContext.__init__
```

````{py:method} _parse_policy_string() -> None
:canonical: src.pipeline.simulations.states.SimulationContext._parse_policy_string

```{autodoc2-docstring} src.pipeline.simulations.states.SimulationContext._parse_policy_string
```

````

````{py:method} _extract_threshold(policy_key: str) -> None
:canonical: src.pipeline.simulations.states.SimulationContext._extract_threshold

```{autodoc2-docstring} src.pipeline.simulations.states.SimulationContext._extract_threshold
```

````

````{py:method} _extract_threshold_with_config_char(policy_key: str, config_chars: typing.List[str]) -> None
:canonical: src.pipeline.simulations.states.SimulationContext._extract_threshold_with_config_char

```{autodoc2-docstring} src.pipeline.simulations.states.SimulationContext._extract_threshold_with_config_char
```

````

````{py:method} _continue_init(variables_dict: typing.Dict[str, typing.Any], pol_id: int) -> None
:canonical: src.pipeline.simulations.states.SimulationContext._continue_init

```{autodoc2-docstring} src.pipeline.simulations.states.SimulationContext._continue_init
```

````

````{py:method} transition_to(state: typing.Optional[src.pipeline.simulations.states.SimState]) -> None
:canonical: src.pipeline.simulations.states.SimulationContext.transition_to

```{autodoc2-docstring} src.pipeline.simulations.states.SimulationContext.transition_to
```

````

````{py:method} run() -> typing.Optional[typing.Dict[str, typing.Any]]
:canonical: src.pipeline.simulations.states.SimulationContext.run

```{autodoc2-docstring} src.pipeline.simulations.states.SimulationContext.run
```

````

````{py:method} get_current_state_tuple() -> typing.Tuple[typing.Any, ...]
:canonical: src.pipeline.simulations.states.SimulationContext.get_current_state_tuple

```{autodoc2-docstring} src.pipeline.simulations.states.SimulationContext.get_current_state_tuple
```

````

`````

`````{py:class} SimState
:canonical: src.pipeline.simulations.states.SimState

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.simulations.states.SimState
```

````{py:attribute} context
:canonical: src.pipeline.simulations.states.SimState.context
:type: src.pipeline.simulations.states.SimulationContext
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.states.SimState.context
```

````

````{py:method} handle(ctx: src.pipeline.simulations.states.SimulationContext) -> None
:canonical: src.pipeline.simulations.states.SimState.handle
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.states.SimState.handle
```

````

`````

`````{py:class} InitializingState
:canonical: src.pipeline.simulations.states.InitializingState

Bases: {py:obj}`src.pipeline.simulations.states.SimState`

```{autodoc2-docstring} src.pipeline.simulations.states.InitializingState
```

````{py:method} handle(ctx: src.pipeline.simulations.states.SimulationContext) -> None
:canonical: src.pipeline.simulations.states.InitializingState.handle

```{autodoc2-docstring} src.pipeline.simulations.states.InitializingState.handle
```

````

`````

`````{py:class} RunningState
:canonical: src.pipeline.simulations.states.RunningState

Bases: {py:obj}`src.pipeline.simulations.states.SimState`

```{autodoc2-docstring} src.pipeline.simulations.states.RunningState
```

````{py:method} handle(ctx: src.pipeline.simulations.states.SimulationContext) -> None
:canonical: src.pipeline.simulations.states.RunningState.handle

```{autodoc2-docstring} src.pipeline.simulations.states.RunningState.handle
```

````

`````

`````{py:class} FinishingState
:canonical: src.pipeline.simulations.states.FinishingState

Bases: {py:obj}`src.pipeline.simulations.states.SimState`

```{autodoc2-docstring} src.pipeline.simulations.states.FinishingState
```

````{py:method} handle(ctx: src.pipeline.simulations.states.SimulationContext) -> None
:canonical: src.pipeline.simulations.states.FinishingState.handle

```{autodoc2-docstring} src.pipeline.simulations.states.FinishingState.handle
```

````

`````
