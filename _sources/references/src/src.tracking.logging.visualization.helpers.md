# {py:mod}`src.tracking.logging.visualization.helpers`

```{py:module} src.tracking.logging.visualization.helpers
```

```{autodoc2-docstring} src.tracking.logging.visualization.helpers
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MyModelWrapper <src.tracking.logging.visualization.helpers.MyModelWrapper>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.helpers.MyModelWrapper
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_batch <src.tracking.logging.visualization.helpers.get_batch>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.helpers.get_batch
    :summary:
    ```
* - {py:obj}`load_model_instance <src.tracking.logging.visualization.helpers.load_model_instance>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.helpers.load_model_instance
    :summary:
    ```
````

### API

````{py:function} get_batch(device, size=50, batch_size=32, temporal_horizon=0)
:canonical: src.tracking.logging.visualization.helpers.get_batch

```{autodoc2-docstring} src.tracking.logging.visualization.helpers.get_batch
```
````

`````{py:class} MyModelWrapper(model)
:canonical: src.tracking.logging.visualization.helpers.MyModelWrapper

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.tracking.logging.visualization.helpers.MyModelWrapper
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.logging.visualization.helpers.MyModelWrapper.__init__
```

````{py:method} forward(input, cost_weights=None, return_pi=False, pad=False, mask=None, expert_pi=None)
:canonical: src.tracking.logging.visualization.helpers.MyModelWrapper.forward

```{autodoc2-docstring} src.tracking.logging.visualization.helpers.MyModelWrapper.forward
```

````

`````

````{py:function} load_model_instance(model_path, device, size=100, problem_name='wcvrp')
:canonical: src.tracking.logging.visualization.helpers.load_model_instance

```{autodoc2-docstring} src.tracking.logging.visualization.helpers.load_model_instance
```
````
