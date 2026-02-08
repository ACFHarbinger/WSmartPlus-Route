# {py:mod}`src.models.critic_network.model`

```{py:module} src.models.critic_network.model
```

```{autodoc2-docstring} src.models.critic_network.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LegacyCriticNetwork <src.models.critic_network.model.LegacyCriticNetwork>`
  - ```{autodoc2-docstring} src.models.critic_network.model.LegacyCriticNetwork
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.models.critic_network.model.__all__>`
  - ```{autodoc2-docstring} src.models.critic_network.model.__all__
    :summary:
    ```
````

### API

`````{py:class} LegacyCriticNetwork(problem, component_factory, embed_dim, hidden_dim, n_layers, n_sublayers, encoder_normalization='batch', activation='gelu', n_heads=8, aggregation_graph='avg', dropout_rate=0.0, temporal_horizon=0)
:canonical: src.models.critic_network.model.LegacyCriticNetwork

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.critic_network.model.LegacyCriticNetwork
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.critic_network.model.LegacyCriticNetwork.__init__
```

````{py:method} _init_embed(nodes)
:canonical: src.models.critic_network.model.LegacyCriticNetwork._init_embed

```{autodoc2-docstring} src.models.critic_network.model.LegacyCriticNetwork._init_embed
```

````

````{py:method} forward(inputs)
:canonical: src.models.critic_network.model.LegacyCriticNetwork.forward

```{autodoc2-docstring} src.models.critic_network.model.LegacyCriticNetwork.forward
```

````

`````

````{py:data} __all__
:canonical: src.models.critic_network.model.__all__
:value: >
   ['CriticNetwork', 'LegacyCriticNetwork', 'create_critic_from_actor']

```{autodoc2-docstring} src.models.critic_network.model.__all__
```

````
