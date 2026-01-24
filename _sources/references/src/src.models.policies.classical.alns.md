# {py:mod}`src.models.policies.classical.alns`

```{py:module} src.models.policies.classical.alns
```

```{autodoc2-docstring} src.models.policies.classical.alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ALNSPolicy <src.models.policies.classical.alns.ALNSPolicy>`
  - ```{autodoc2-docstring} src.models.policies.classical.alns.ALNSPolicy
    :summary:
    ```
````

### API

`````{py:class} ALNSPolicy(env_name: str, time_limit: float = 5.0, max_iterations: int = 100, **kwargs)
:canonical: src.models.policies.classical.alns.ALNSPolicy

Bases: {py:obj}`logic.src.models.policies.base.ConstructivePolicy`

```{autodoc2-docstring} src.models.policies.classical.alns.ALNSPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.classical.alns.ALNSPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'greedy', **kwargs) -> dict
:canonical: src.models.policies.classical.alns.ALNSPolicy.forward

```{autodoc2-docstring} src.models.policies.classical.alns.ALNSPolicy.forward
```

````

`````
