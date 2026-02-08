# {py:mod}`src.models.hrl_manager.manager`

```{py:module} src.models.hrl_manager.manager
```

```{autodoc2-docstring} src.models.hrl_manager.manager
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GATLSTManager <src.models.hrl_manager.manager.GATLSTManager>`
  - ```{autodoc2-docstring} src.models.hrl_manager.manager.GATLSTManager
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.models.hrl_manager.manager.logger>`
  - ```{autodoc2-docstring} src.models.hrl_manager.manager.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.models.hrl_manager.manager.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.models.hrl_manager.manager.logger
```

````

`````{py:class} GATLSTManager(input_dim_static=STATIC_DIM, input_dim_dynamic=DEFAULT_TEMPORAL_HORIZON, global_input_dim=2, critical_threshold=CRITICAL_FILL_THRESHOLD, batch_size=1024, hidden_dim=128, lstm_hidden=64, num_layers_gat=3, n_heads=8, dropout=0.1, device='cuda', shared_encoder=None)
:canonical: src.models.hrl_manager.manager.GATLSTManager

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.hrl_manager.manager.GATLSTManager
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.hrl_manager.manager.GATLSTManager.__init__
```

````{py:method} clear_memory()
:canonical: src.models.hrl_manager.manager.GATLSTManager.clear_memory

```{autodoc2-docstring} src.models.hrl_manager.manager.GATLSTManager.clear_memory
```

````

````{py:method} feature_processing(static, dynamic)
:canonical: src.models.hrl_manager.manager.GATLSTManager.feature_processing

```{autodoc2-docstring} src.models.hrl_manager.manager.GATLSTManager.feature_processing
```

````

````{py:method} forward(static, dynamic, global_features)
:canonical: src.models.hrl_manager.manager.GATLSTManager.forward

```{autodoc2-docstring} src.models.hrl_manager.manager.GATLSTManager.forward
```

````

````{py:method} select_action(static, dynamic, global_features=None, deterministic=False, threshold=0.5, must_go_threshold=0.5, target_must_go=None)
:canonical: src.models.hrl_manager.manager.GATLSTManager.select_action

```{autodoc2-docstring} src.models.hrl_manager.manager.GATLSTManager.select_action
```

````

````{py:method} get_must_go_mask(static, dynamic, global_features=None, threshold=0.5)
:canonical: src.models.hrl_manager.manager.GATLSTManager.get_must_go_mask

```{autodoc2-docstring} src.models.hrl_manager.manager.GATLSTManager.get_must_go_mask
```

````

`````
