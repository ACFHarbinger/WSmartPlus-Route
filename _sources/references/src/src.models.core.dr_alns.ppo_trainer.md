# {py:mod}`src.models.core.dr_alns.ppo_trainer`

```{py:module} src.models.core.dr_alns.ppo_trainer
```

```{autodoc2-docstring} src.models.core.dr_alns.ppo_trainer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PPOBuffer <src.models.core.dr_alns.ppo_trainer.PPOBuffer>`
  - ```{autodoc2-docstring} src.models.core.dr_alns.ppo_trainer.PPOBuffer
    :summary:
    ```
* - {py:obj}`PPOTrainer <src.models.core.dr_alns.ppo_trainer.PPOTrainer>`
  - ```{autodoc2-docstring} src.models.core.dr_alns.ppo_trainer.PPOTrainer
    :summary:
    ```
````

### API

`````{py:class} PPOBuffer(capacity: int = 10000)
:canonical: src.models.core.dr_alns.ppo_trainer.PPOBuffer

```{autodoc2-docstring} src.models.core.dr_alns.ppo_trainer.PPOBuffer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.dr_alns.ppo_trainer.PPOBuffer.__init__
```

````{py:method} reset() -> None
:canonical: src.models.core.dr_alns.ppo_trainer.PPOBuffer.reset

```{autodoc2-docstring} src.models.core.dr_alns.ppo_trainer.PPOBuffer.reset
```

````

````{py:method} add(state: torch.Tensor, action: typing.Dict[str, int], log_prob: typing.Dict[str, torch.Tensor], reward: float, done: bool, value: torch.Tensor)
:canonical: src.models.core.dr_alns.ppo_trainer.PPOBuffer.add

```{autodoc2-docstring} src.models.core.dr_alns.ppo_trainer.PPOBuffer.add
```

````

````{py:method} get(device: torch.device) -> typing.Dict[str, typing.Any]
:canonical: src.models.core.dr_alns.ppo_trainer.PPOBuffer.get

```{autodoc2-docstring} src.models.core.dr_alns.ppo_trainer.PPOBuffer.get
```

````

````{py:method} __len__() -> int
:canonical: src.models.core.dr_alns.ppo_trainer.PPOBuffer.__len__

```{autodoc2-docstring} src.models.core.dr_alns.ppo_trainer.PPOBuffer.__len__
```

````

`````

`````{py:class} PPOTrainer(agent: src.models.core.dr_alns.ppo_agent.DRALNSPPOAgent, env: logic.src.envs.dr_alns.DRALNSEnv, lr: float = 0.0003, gamma: float = 0.99, gae_lambda: float = 0.95, clip_epsilon: float = 0.2, value_loss_coef: float = 0.5, entropy_coef: float = 0.01, max_grad_norm: float = 0.5, n_epochs: int = 10, batch_size: int = 64, device: typing.Optional[torch.device] = None)
:canonical: src.models.core.dr_alns.ppo_trainer.PPOTrainer

```{autodoc2-docstring} src.models.core.dr_alns.ppo_trainer.PPOTrainer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.dr_alns.ppo_trainer.PPOTrainer.__init__
```

````{py:method} collect_experience(n_steps: int, instance_generator: typing.Optional[typing.Callable[..., typing.Any]] = None) -> typing.Dict[str, float]
:canonical: src.models.core.dr_alns.ppo_trainer.PPOTrainer.collect_experience

```{autodoc2-docstring} src.models.core.dr_alns.ppo_trainer.PPOTrainer.collect_experience
```

````

````{py:method} compute_advantages(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.core.dr_alns.ppo_trainer.PPOTrainer.compute_advantages

```{autodoc2-docstring} src.models.core.dr_alns.ppo_trainer.PPOTrainer.compute_advantages
```

````

````{py:method} update() -> typing.Dict[str, float]
:canonical: src.models.core.dr_alns.ppo_trainer.PPOTrainer.update

```{autodoc2-docstring} src.models.core.dr_alns.ppo_trainer.PPOTrainer.update
```

````

````{py:method} train(total_timesteps: int, n_steps_per_update: int = 2048, log_interval: int = 10, instance_generator: typing.Optional[typing.Callable] = None) -> typing.Dict[str, typing.List[float]]
:canonical: src.models.core.dr_alns.ppo_trainer.PPOTrainer.train

```{autodoc2-docstring} src.models.core.dr_alns.ppo_trainer.PPOTrainer.train
```

````

`````
