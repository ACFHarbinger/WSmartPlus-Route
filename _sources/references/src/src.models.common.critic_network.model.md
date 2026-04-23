# {py:mod}`src.models.common.critic_network.model`

```{py:module} src.models.common.critic_network.model
```

```{autodoc2-docstring} src.models.common.critic_network.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LegacyCriticNetwork <src.models.common.critic_network.model.LegacyCriticNetwork>`
  - ```{autodoc2-docstring} src.models.common.critic_network.model.LegacyCriticNetwork
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.models.common.critic_network.model.__all__>`
  - ```{autodoc2-docstring} src.models.common.critic_network.model.__all__
    :summary:
    ```
````

### API

`````{py:class} LegacyCriticNetwork(problem: typing.Any, component_factory: typing.Any, embed_dim: int, hidden_dim: int, n_layers: int, n_sublayers: int, encoder_normalization: str = 'batch', activation: str = 'gelu', n_heads: int = 8, aggregation_graph: str = 'avg', dropout_rate: float = 0.0, temporal_horizon: int = 0)
:canonical: src.models.common.critic_network.model.LegacyCriticNetwork

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.common.critic_network.model.LegacyCriticNetwork
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.critic_network.model.LegacyCriticNetwork.__init__
```

````{py:method} _init_embed(nodes: torch.Tensor) -> torch.Tensor
:canonical: src.models.common.critic_network.model.LegacyCriticNetwork._init_embed

```{autodoc2-docstring} src.models.common.critic_network.model.LegacyCriticNetwork._init_embed
```

````

````{py:method} forward(inputs: typing.Dict[str, typing.Any]) -> torch.Tensor
:canonical: src.models.common.critic_network.model.LegacyCriticNetwork.forward

```{autodoc2-docstring} src.models.common.critic_network.model.LegacyCriticNetwork.forward
```

````

`````

````{py:data} __all__
:canonical: src.models.common.critic_network.model.__all__
:value: >
   ['CriticNetwork', 'LegacyCriticNetwork', 'create_critic_from_actor']

```{autodoc2-docstring} src.models.common.critic_network.model.__all__
```

````
