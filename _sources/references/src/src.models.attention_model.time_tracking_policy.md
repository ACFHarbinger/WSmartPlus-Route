# {py:mod}`src.models.attention_model.time_tracking_policy`

```{py:module} src.models.attention_model.time_tracking_policy
```

```{autodoc2-docstring} src.models.attention_model.time_tracking_policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TimeTrackingPolicy <src.models.attention_model.time_tracking_policy.TimeTrackingPolicy>`
  - ```{autodoc2-docstring} src.models.attention_model.time_tracking_policy.TimeTrackingPolicy
    :summary:
    ```
````

### API

`````{py:class} TimeTrackingPolicy(policy: torch.nn.Module)
:canonical: src.models.attention_model.time_tracking_policy.TimeTrackingPolicy

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.attention_model.time_tracking_policy.TimeTrackingPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.attention_model.time_tracking_policy.TimeTrackingPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.attention_model.time_tracking_policy.TimeTrackingPolicy.forward

```{autodoc2-docstring} src.models.attention_model.time_tracking_policy.TimeTrackingPolicy.forward
```

````

````{py:method} __getattr__(name: str) -> typing.Any
:canonical: src.models.attention_model.time_tracking_policy.TimeTrackingPolicy.__getattr__

```{autodoc2-docstring} src.models.attention_model.time_tracking_policy.TimeTrackingPolicy.__getattr__
```

````

`````
