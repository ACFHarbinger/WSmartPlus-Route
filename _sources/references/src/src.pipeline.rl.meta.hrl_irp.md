# {py:mod}`src.pipeline.rl.meta.hrl_irp`

```{py:module} src.pipeline.rl.meta.hrl_irp
```

```{autodoc2-docstring} src.pipeline.rl.meta.hrl_irp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HRLIRPModule <src.pipeline.rl.meta.hrl_irp.HRLIRPModule>`
  - ```{autodoc2-docstring} src.pipeline.rl.meta.hrl_irp.HRLIRPModule
    :summary:
    ```
````

### API

`````{py:class} HRLIRPModule(horizon: int = 7, gamma: float = 0.99, alpha_overflow: float = 1.0, overflow_penalty: float = 500.0, ppo_epochs: int = 4, entropy_coef: float = 0.01, **kwargs: typing.Any)
:canonical: src.pipeline.rl.meta.hrl_irp.HRLIRPModule

Bases: {py:obj}`logic.src.pipeline.rl.meta.hrl.HRLModule`

```{autodoc2-docstring} src.pipeline.rl.meta.hrl_irp.HRLIRPModule
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.meta.hrl_irp.HRLIRPModule.__init__
```

````{py:method} _compute_day_reward(actions: torch.Tensor, routing_cost: torch.Tensor, fill_levels: torch.Tensor, bin_capacity: float = 100.0) -> torch.Tensor
:canonical: src.pipeline.rl.meta.hrl_irp.HRLIRPModule._compute_day_reward

```{autodoc2-docstring} src.pipeline.rl.meta.hrl_irp.HRLIRPModule._compute_day_reward
```

````

````{py:method} _simulate_inventory_transition(fill_levels: torch.Tensor, actions: torch.Tensor, demand_delta: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.pipeline.rl.meta.hrl_irp.HRLIRPModule._simulate_inventory_transition

```{autodoc2-docstring} src.pipeline.rl.meta.hrl_irp.HRLIRPModule._simulate_inventory_transition
```

````

````{py:method} _collect_horizon(td: tensordict.TensorDict) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.List[torch.Tensor], typing.List[torch.Tensor]]
:canonical: src.pipeline.rl.meta.hrl_irp.HRLIRPModule._collect_horizon

```{autodoc2-docstring} src.pipeline.rl.meta.hrl_irp.HRLIRPModule._collect_horizon
```

````

````{py:method} training_step(batch: tensordict.TensorDict, batch_idx: int) -> None
:canonical: src.pipeline.rl.meta.hrl_irp.HRLIRPModule.training_step

```{autodoc2-docstring} src.pipeline.rl.meta.hrl_irp.HRLIRPModule.training_step
```

````

`````
