# {py:mod}`src.models.policies.critic`

```{py:module} src.models.policies.critic
```

```{autodoc2-docstring} src.models.policies.critic
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CriticNetwork <src.models.policies.critic.CriticNetwork>`
  - ```{autodoc2-docstring} src.models.policies.critic.CriticNetwork
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_critic_from_actor <src.models.policies.critic.create_critic_from_actor>`
  - ```{autodoc2-docstring} src.models.policies.critic.create_critic_from_actor
    :summary:
    ```
````

### API

`````{py:class} CriticNetwork(env_name: str, embed_dim: int = 128, hidden_dim: int = 128, n_layers: int = 3, n_heads: int = 8, normalization: str = 'batch', dropout_rate: float = 0.0, aggregation: str = 'avg', encoder: typing.Optional[torch.nn.Module] = None, **kwargs)
:canonical: src.models.policies.critic.CriticNetwork

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.policies.critic.CriticNetwork
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.critic.CriticNetwork.__init__
```

````{py:method} forward(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.models.policies.critic.CriticNetwork.forward

```{autodoc2-docstring} src.models.policies.critic.CriticNetwork.forward
```

````

`````

````{py:function} create_critic_from_actor(policy: torch.nn.Module, backbone_name: str = 'encoder', **critic_kwargs)
:canonical: src.models.policies.critic.create_critic_from_actor

```{autodoc2-docstring} src.models.policies.critic.create_critic_from_actor
```
````
