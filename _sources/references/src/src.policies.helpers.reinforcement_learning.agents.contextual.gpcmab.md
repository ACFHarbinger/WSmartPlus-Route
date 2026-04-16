# {py:mod}`src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab`

```{py:module} src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab
```

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GPCMABAgent <src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent>`
  - ```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent
    :summary:
    ```
````

### API

`````{py:class} GPCMABAgent(n_arms: int, feature_dim: int, beta: float = 2.0, length_scale: float = 1.0, signal_variance: float = 1.0, noise_variance: float = 0.1, max_history: int = 500, super_arm_size: int = 1, seed: typing.Optional[int] = None, history_size: int = 50)
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent

Bases: {py:obj}`src.policies.helpers.reinforcement_learning.agents.contextual.base.ContextualBanditAgent`

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent.__init__
```

````{py:method} _encode_input(context: numpy.ndarray, action: int) -> numpy.ndarray
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent._encode_input

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent._encode_input
```

````

````{py:method} _rbf_kernel(x1: numpy.ndarray, x2: numpy.ndarray) -> float
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent._rbf_kernel

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent._rbf_kernel
```

````

````{py:method} _kernel_vector(x_star: numpy.ndarray) -> numpy.ndarray
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent._kernel_vector

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent._kernel_vector
```

````

````{py:method} _predict(context: numpy.ndarray, action: int) -> typing.Tuple[float, float]
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent._predict

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent._predict
```

````

````{py:method} _recompute_kernel_inverse() -> None
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent._recompute_kernel_inverse

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent._recompute_kernel_inverse
```

````

````{py:method} select_action(context: numpy.ndarray, rng: numpy.random.Generator) -> int
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent.select_action

````

````{py:method} select_super_arm(context: numpy.ndarray, rng: numpy.random.Generator) -> typing.List[int]
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent.select_super_arm

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent.select_super_arm
```

````

````{py:method} update(context: numpy.ndarray, action: int, reward: float, next_context: typing.Any = None, done: bool = False) -> None
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent.update

````

````{py:method} get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent.get_statistics

````

````{py:method} get_weights() -> typing.Optional[numpy.ndarray]
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent.get_weights

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent.get_weights
```

````

````{py:method} save(path: str) -> None
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent.save

````

````{py:method} load(path: str) -> None
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent.load

````

````{py:method} reset() -> None
:canonical: src.policies.helpers.reinforcement_learning.agents.contextual.gpcmab.GPCMABAgent.reset

````

`````
