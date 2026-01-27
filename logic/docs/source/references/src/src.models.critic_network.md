# {py:mod}`src.models.critic_network`

```{py:module} src.models.critic_network
```

```{autodoc2-docstring} src.models.critic_network
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CriticNetwork <src.models.critic_network.CriticNetwork>`
  - ```{autodoc2-docstring} src.models.critic_network.CriticNetwork
    :summary:
    ```
````

### API

`````{py:class} CriticNetwork(problem, component_factory, embedding_dim, hidden_dim, n_layers, n_sublayers, encoder_normalization='batch', activation='gelu', n_heads=8, aggregation_graph='avg', dropout_rate=0.0, temporal_horizon=0)
:canonical: src.models.critic_network.CriticNetwork

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.critic_network.CriticNetwork
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.critic_network.CriticNetwork.__init__
```

````{py:method} _init_embed(nodes)
:canonical: src.models.critic_network.CriticNetwork._init_embed

```{autodoc2-docstring} src.models.critic_network.CriticNetwork._init_embed
```

````

````{py:method} forward(inputs)
:canonical: src.models.critic_network.CriticNetwork.forward

```{autodoc2-docstring} src.models.critic_network.CriticNetwork.forward
```

````

`````
