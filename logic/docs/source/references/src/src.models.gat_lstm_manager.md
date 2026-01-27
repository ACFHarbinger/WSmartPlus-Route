# {py:mod}`src.models.gat_lstm_manager`

```{py:module} src.models.gat_lstm_manager
```

```{autodoc2-docstring} src.models.gat_lstm_manager
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GATLSTManager <src.models.gat_lstm_manager.GATLSTManager>`
  - ```{autodoc2-docstring} src.models.gat_lstm_manager.GATLSTManager
    :summary:
    ```
````

### API

`````{py:class} GATLSTManager(input_dim_static=2, input_dim_dynamic=10, global_input_dim=2, critical_threshold=0.9, batch_size=1024, hidden_dim=128, lstm_hidden=64, num_layers_gat=3, num_heads=8, dropout=0.1, device='cuda', shared_encoder=None)
:canonical: src.models.gat_lstm_manager.GATLSTManager

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.gat_lstm_manager.GATLSTManager
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.gat_lstm_manager.GATLSTManager.__init__
```

````{py:method} clear_memory()
:canonical: src.models.gat_lstm_manager.GATLSTManager.clear_memory

```{autodoc2-docstring} src.models.gat_lstm_manager.GATLSTManager.clear_memory
```

````

````{py:method} feature_processing(static, dynamic)
:canonical: src.models.gat_lstm_manager.GATLSTManager.feature_processing

```{autodoc2-docstring} src.models.gat_lstm_manager.GATLSTManager.feature_processing
```

````

````{py:method} forward(static, dynamic, global_features)
:canonical: src.models.gat_lstm_manager.GATLSTManager.forward

```{autodoc2-docstring} src.models.gat_lstm_manager.GATLSTManager.forward
```

````

````{py:method} select_action(static, dynamic, global_features=None, deterministic=False, threshold=0.5, mask_threshold=0.5, target_mask=None)
:canonical: src.models.gat_lstm_manager.GATLSTManager.select_action

```{autodoc2-docstring} src.models.gat_lstm_manager.GATLSTManager.select_action
```

````

`````
