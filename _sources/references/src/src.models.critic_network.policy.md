# {py:mod}`src.models.critic_network.policy`

```{py:module} src.models.critic_network.policy
```

```{autodoc2-docstring} src.models.critic_network.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CriticNetwork <src.models.critic_network.policy.CriticNetwork>`
  - ```{autodoc2-docstring} src.models.critic_network.policy.CriticNetwork
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_critic_from_actor <src.models.critic_network.policy.create_critic_from_actor>`
  - ```{autodoc2-docstring} src.models.critic_network.policy.create_critic_from_actor
    :summary:
    ```
````

### API

`````{py:class} CriticNetwork(env_name: str, embed_dim: int = 128, hidden_dim: int = 128, n_layers: int = 3, n_heads: int = 8, normalization: str = 'batch', dropout_rate: float = 0.0, aggregation: str = 'avg', encoder: typing.Optional[torch.nn.Module] = None, **kwargs)
:canonical: src.models.critic_network.policy.CriticNetwork

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.critic_network.policy.CriticNetwork
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.critic_network.policy.CriticNetwork.__init__
```

````{py:method} forward(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.models.critic_network.policy.CriticNetwork.forward

```{autodoc2-docstring} src.models.critic_network.policy.CriticNetwork.forward
```

````

`````

````{py:function} create_critic_from_actor(policy: torch.nn.Module, backbone_name: str = 'encoder', **critic_kwargs)
:canonical: src.models.critic_network.policy.create_critic_from_actor

```{autodoc2-docstring} src.models.critic_network.policy.create_critic_from_actor
```
````
