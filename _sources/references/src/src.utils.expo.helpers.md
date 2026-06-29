# {py:mod}`src.utils.expo.helpers`

```{py:module} src.utils.expo.helpers
```

```{autodoc2-docstring} src.utils.expo.helpers
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MyModelWrapper <src.utils.expo.helpers.MyModelWrapper>`
  - ```{autodoc2-docstring} src.utils.expo.helpers.MyModelWrapper
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_batch <src.utils.expo.helpers.get_batch>`
  - ```{autodoc2-docstring} src.utils.expo.helpers.get_batch
    :summary:
    ```
* - {py:obj}`load_model_instance <src.utils.expo.helpers.load_model_instance>`
  - ```{autodoc2-docstring} src.utils.expo.helpers.load_model_instance
    :summary:
    ```
````

### API

````{py:function} get_batch(device, size=50, batch_size=32, temporal_horizon=0)
:canonical: src.utils.expo.helpers.get_batch

```{autodoc2-docstring} src.utils.expo.helpers.get_batch
```
````

`````{py:class} MyModelWrapper(model)
:canonical: src.utils.expo.helpers.MyModelWrapper

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.utils.expo.helpers.MyModelWrapper
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.expo.helpers.MyModelWrapper.__init__
```

````{py:method} forward(input, cost_weights=None, return_pi=False, pad=False, mask=None, expert_pi=None)
:canonical: src.utils.expo.helpers.MyModelWrapper.forward

```{autodoc2-docstring} src.utils.expo.helpers.MyModelWrapper.forward
```

````

`````

````{py:function} load_model_instance(model_path, device, size=100, problem_name='wcvrp')
:canonical: src.utils.expo.helpers.load_model_instance

```{autodoc2-docstring} src.utils.expo.helpers.load_model_instance
```
````
