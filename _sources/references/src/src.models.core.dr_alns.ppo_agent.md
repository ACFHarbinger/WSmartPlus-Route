# {py:mod}`src.models.core.dr_alns.ppo_agent`

```{py:module} src.models.core.dr_alns.ppo_agent
```

```{autodoc2-docstring} src.models.core.dr_alns.ppo_agent
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DRALNSPPOAgent <src.models.core.dr_alns.ppo_agent.DRALNSPPOAgent>`
  - ```{autodoc2-docstring} src.models.core.dr_alns.ppo_agent.DRALNSPPOAgent
    :summary:
    ```
* - {py:obj}`DRALNSState <src.models.core.dr_alns.ppo_agent.DRALNSState>`
  - ```{autodoc2-docstring} src.models.core.dr_alns.ppo_agent.DRALNSState
    :summary:
    ```
````

### API

`````{py:class} DRALNSPPOAgent(state_dim: int = 7, hidden_dim: int = 64, n_destroy_ops: int = 3, n_repair_ops: int = 3, n_severity_levels: int = 10, n_temp_levels: int = 50)
:canonical: src.models.core.dr_alns.ppo_agent.DRALNSPPOAgent

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.core.dr_alns.ppo_agent.DRALNSPPOAgent
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.dr_alns.ppo_agent.DRALNSPPOAgent.__init__
```

````{py:method} forward(state: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.models.core.dr_alns.ppo_agent.DRALNSPPOAgent.forward

```{autodoc2-docstring} src.models.core.dr_alns.ppo_agent.DRALNSPPOAgent.forward
```

````

````{py:method} get_action(state: torch.Tensor, deterministic: bool = False) -> typing.Tuple[typing.Dict[str, int], typing.Dict[str, torch.Tensor], torch.Tensor]
:canonical: src.models.core.dr_alns.ppo_agent.DRALNSPPOAgent.get_action

```{autodoc2-docstring} src.models.core.dr_alns.ppo_agent.DRALNSPPOAgent.get_action
```

````

````{py:method} evaluate_actions(states: torch.Tensor, actions: typing.Dict[str, torch.Tensor]) -> typing.Tuple[typing.Dict[str, torch.Tensor], torch.Tensor, typing.Dict[str, torch.Tensor]]
:canonical: src.models.core.dr_alns.ppo_agent.DRALNSPPOAgent.evaluate_actions

```{autodoc2-docstring} src.models.core.dr_alns.ppo_agent.DRALNSPPOAgent.evaluate_actions
```

````

`````

`````{py:class} DRALNSState()
:canonical: src.models.core.dr_alns.ppo_agent.DRALNSState

```{autodoc2-docstring} src.models.core.dr_alns.ppo_agent.DRALNSState
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.dr_alns.ppo_agent.DRALNSState.__init__
```

````{py:method} to_tensor(device: typing.Optional[torch.device] = None) -> torch.Tensor
:canonical: src.models.core.dr_alns.ppo_agent.DRALNSState.to_tensor

```{autodoc2-docstring} src.models.core.dr_alns.ppo_agent.DRALNSState.to_tensor
```

````

````{py:method} update(best_profit: float, current_profit: float, previous_profit: float, new_accepted: bool, new_best: bool, iteration: int, max_iterations: int, iterations_since_best: int)
:canonical: src.models.core.dr_alns.ppo_agent.DRALNSState.update

```{autodoc2-docstring} src.models.core.dr_alns.ppo_agent.DRALNSState.update
```

````

`````
