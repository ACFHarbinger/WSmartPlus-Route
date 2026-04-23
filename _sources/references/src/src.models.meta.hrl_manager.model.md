# {py:mod}`src.models.meta.hrl_manager.model`

```{py:module} src.models.meta.hrl_manager.model
```

```{autodoc2-docstring} src.models.meta.hrl_manager.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MandatoryManager <src.models.meta.hrl_manager.model.MandatoryManager>`
  - ```{autodoc2-docstring} src.models.meta.hrl_manager.model.MandatoryManager
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.models.meta.hrl_manager.model.logger>`
  - ```{autodoc2-docstring} src.models.meta.hrl_manager.model.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.models.meta.hrl_manager.model.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.models.meta.hrl_manager.model.logger
```

````

`````{py:class} MandatoryManager(input_dim_static: int = STATIC_DIM, input_dim_dynamic: int = DEFAULT_TEMPORAL_HORIZON, global_input_dim: int = 2, critical_threshold: float = CRITICAL_FILL_THRESHOLD, batch_size: int = DEFAULT_EVAL_BATCH_SIZE, hidden_dim: int = 128, embed_dim: typing.Optional[int] = None, feed_forward_hidden: typing.Optional[int] = None, lstm_hidden: int = 64, num_layers_gat: int = 3, n_heads: int = 8, dropout: float = 0.1, device: str = 'cuda', shared_encoder: typing.Optional[torch.nn.Module] = None, temporal_encoder_cls: typing.Optional[typing.Type[torch.nn.Module]] = None, temporal_encoder_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, spatial_encoder_cls: typing.Optional[typing.Type[torch.nn.Module]] = None, spatial_encoder_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, component_factory: typing.Optional[logic.src.models.subnets.factories.NeuralComponentFactory] = None, temporal_encoder_type: str = 'lstm', norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None)
:canonical: src.models.meta.hrl_manager.model.MandatoryManager

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.meta.hrl_manager.model.MandatoryManager
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.meta.hrl_manager.model.MandatoryManager.__init__
```

````{py:method} _init_shared_encoder_dim(shared_encoder: typing.Optional[torch.nn.Module], embed_dim: int, hidden_dim: int) -> typing.Tuple[int, int]
:canonical: src.models.meta.hrl_manager.model.MandatoryManager._init_shared_encoder_dim

```{autodoc2-docstring} src.models.meta.hrl_manager.model.MandatoryManager._init_shared_encoder_dim
```

````

````{py:method} _init_temporal_encoder(temporal_encoder_cls: typing.Optional[typing.Type[torch.nn.Module]] = None, temporal_encoder_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, lstm_hidden: int = 64, temporal_encoder_type: str = 'lstm') -> torch.nn.Module
:canonical: src.models.meta.hrl_manager.model.MandatoryManager._init_temporal_encoder

```{autodoc2-docstring} src.models.meta.hrl_manager.model.MandatoryManager._init_temporal_encoder
```

````

````{py:method} _init_spatial_encoder(shared_encoder: typing.Optional[torch.nn.Module] = None, component_factory: typing.Optional[logic.src.models.subnets.factories.NeuralComponentFactory] = None, spatial_encoder_cls: typing.Optional[typing.Type[torch.nn.Module]] = None, spatial_encoder_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, embed_dim: int = 128, feed_forward_hidden: int = 512, num_layers_gat: int = 3, n_heads: int = 8, dropout: float = 0.1, norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None) -> torch.nn.Module
:canonical: src.models.meta.hrl_manager.model.MandatoryManager._init_spatial_encoder

```{autodoc2-docstring} src.models.meta.hrl_manager.model.MandatoryManager._init_spatial_encoder
```

````

````{py:method} _init_heads(hidden_dim: int, lstm_hidden: int, global_input_dim: int) -> None
:canonical: src.models.meta.hrl_manager.model.MandatoryManager._init_heads

```{autodoc2-docstring} src.models.meta.hrl_manager.model.MandatoryManager._init_heads
```

````

````{py:method} _initialize_head_weights() -> None
:canonical: src.models.meta.hrl_manager.model.MandatoryManager._initialize_head_weights

```{autodoc2-docstring} src.models.meta.hrl_manager.model.MandatoryManager._initialize_head_weights
```

````

````{py:method} clear_memory() -> None
:canonical: src.models.meta.hrl_manager.model.MandatoryManager.clear_memory

```{autodoc2-docstring} src.models.meta.hrl_manager.model.MandatoryManager.clear_memory
```

````

````{py:method} feature_processing(static: torch.Tensor, dynamic: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.meta.hrl_manager.model.MandatoryManager.feature_processing

```{autodoc2-docstring} src.models.meta.hrl_manager.model.MandatoryManager.feature_processing
```

````

````{py:method} forward(static: torch.Tensor, dynamic: torch.Tensor, global_features: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.models.meta.hrl_manager.model.MandatoryManager.forward

```{autodoc2-docstring} src.models.meta.hrl_manager.model.MandatoryManager.forward
```

````

````{py:method} select_action(static: torch.Tensor, dynamic: torch.Tensor, global_features: typing.Optional[torch.Tensor] = None, deterministic: bool = False, threshold: float = 0.5, mandatory_threshold: float = 0.5, target_mandatory: typing.Optional[torch.Tensor] = None) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.models.meta.hrl_manager.model.MandatoryManager.select_action

```{autodoc2-docstring} src.models.meta.hrl_manager.model.MandatoryManager.select_action
```

````

````{py:method} get_mandatory_mask(static: torch.Tensor, dynamic: torch.Tensor, global_features: typing.Optional[torch.Tensor] = None, threshold: float = 0.5) -> torch.Tensor
:canonical: src.models.meta.hrl_manager.model.MandatoryManager.get_mandatory_mask

```{autodoc2-docstring} src.models.meta.hrl_manager.model.MandatoryManager.get_mandatory_mask
```

````

`````
