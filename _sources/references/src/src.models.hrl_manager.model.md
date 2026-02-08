# {py:mod}`src.models.hrl_manager.model`

```{py:module} src.models.hrl_manager.model
```

```{autodoc2-docstring} src.models.hrl_manager.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GATLSTManager <src.models.hrl_manager.model.GATLSTManager>`
  - ```{autodoc2-docstring} src.models.hrl_manager.model.GATLSTManager
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.models.hrl_manager.model.logger>`
  - ```{autodoc2-docstring} src.models.hrl_manager.model.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.models.hrl_manager.model.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.models.hrl_manager.model.logger
```

````

`````{py:class} GATLSTManager(input_dim_static=STATIC_DIM, input_dim_dynamic=DEFAULT_TEMPORAL_HORIZON, global_input_dim=2, critical_threshold=CRITICAL_FILL_THRESHOLD, batch_size=DEFAULT_EVAL_BATCH_SIZE, hidden_dim=128, lstm_hidden=64, num_layers_gat=3, n_heads=8, dropout=0.1, device='cuda', shared_encoder=None, temporal_encoder_cls=None, temporal_encoder_kwargs=None, spatial_encoder_cls=None, spatial_encoder_kwargs=None, component_factory: logic.src.models.subnets.factories.NeuralComponentFactory = None, temporal_encoder_type: str = 'lstm')
:canonical: src.models.hrl_manager.model.GATLSTManager

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.hrl_manager.model.GATLSTManager
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.hrl_manager.model.GATLSTManager.__init__
```

````{py:method} clear_memory()
:canonical: src.models.hrl_manager.model.GATLSTManager.clear_memory

```{autodoc2-docstring} src.models.hrl_manager.model.GATLSTManager.clear_memory
```

````

````{py:method} feature_processing(static, dynamic)
:canonical: src.models.hrl_manager.model.GATLSTManager.feature_processing

```{autodoc2-docstring} src.models.hrl_manager.model.GATLSTManager.feature_processing
```

````

````{py:method} forward(static, dynamic, global_features)
:canonical: src.models.hrl_manager.model.GATLSTManager.forward

```{autodoc2-docstring} src.models.hrl_manager.model.GATLSTManager.forward
```

````

````{py:method} select_action(static, dynamic, global_features=None, deterministic=False, threshold=0.5, must_go_threshold=0.5, target_must_go=None)
:canonical: src.models.hrl_manager.model.GATLSTManager.select_action

```{autodoc2-docstring} src.models.hrl_manager.model.GATLSTManager.select_action
```

````

````{py:method} get_must_go_mask(static, dynamic, global_features=None, threshold=0.5)
:canonical: src.models.hrl_manager.model.GATLSTManager.get_must_go_mask

```{autodoc2-docstring} src.models.hrl_manager.model.GATLSTManager.get_must_go_mask
```

````

`````
