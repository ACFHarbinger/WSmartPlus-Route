# {py:mod}`src.utils.expo.landscape`

```{py:module} src.utils.expo.landscape
```

```{autodoc2-docstring} src.utils.expo.landscape
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImitationMetric <src.utils.expo.landscape.ImitationMetric>`
  - ```{autodoc2-docstring} src.utils.expo.landscape.ImitationMetric
    :summary:
    ```
* - {py:obj}`RLMetric <src.utils.expo.landscape.RLMetric>`
  - ```{autodoc2-docstring} src.utils.expo.landscape.RLMetric
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`imitation_loss_fn <src.utils.expo.landscape.imitation_loss_fn>`
  - ```{autodoc2-docstring} src.utils.expo.landscape.imitation_loss_fn
    :summary:
    ```
* - {py:obj}`rl_loss_fn <src.utils.expo.landscape.rl_loss_fn>`
  - ```{autodoc2-docstring} src.utils.expo.landscape.rl_loss_fn
    :summary:
    ```
* - {py:obj}`plot_loss_landscape <src.utils.expo.landscape.plot_loss_landscape>`
  - ```{autodoc2-docstring} src.utils.expo.landscape.plot_loss_landscape
    :summary:
    ```
````

### API

````{py:function} imitation_loss_fn(m, x_batch, pi_target, cost_weights=None)
:canonical: src.utils.expo.landscape.imitation_loss_fn

```{autodoc2-docstring} src.utils.expo.landscape.imitation_loss_fn
```
````

````{py:function} rl_loss_fn(m, x_batch, cost_weights=None)
:canonical: src.utils.expo.landscape.rl_loss_fn

```{autodoc2-docstring} src.utils.expo.landscape.rl_loss_fn
```
````

`````{py:class} ImitationMetric(x_batch, pi_target, cost_weights=None)
:canonical: src.utils.expo.landscape.ImitationMetric

Bases: {py:obj}`loss_landscapes.metrics.Metric`

```{autodoc2-docstring} src.utils.expo.landscape.ImitationMetric
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.expo.landscape.ImitationMetric.__init__
```

````{py:method} __call__(model_wrapper)
:canonical: src.utils.expo.landscape.ImitationMetric.__call__

```{autodoc2-docstring} src.utils.expo.landscape.ImitationMetric.__call__
```

````

`````

`````{py:class} RLMetric(x_batch, cost_weights=None)
:canonical: src.utils.expo.landscape.RLMetric

Bases: {py:obj}`loss_landscapes.metrics.Metric`

```{autodoc2-docstring} src.utils.expo.landscape.RLMetric
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.expo.landscape.RLMetric.__init__
```

````{py:method} __call__(model_wrapper)
:canonical: src.utils.expo.landscape.RLMetric.__call__

```{autodoc2-docstring} src.utils.expo.landscape.RLMetric.__call__
```

````

`````

````{py:function} plot_loss_landscape(model: typing.Any, cfg: typing.Union[logic.src.configs.Config, omegaconf.DictConfig], output_dir: str, epoch: int = 0, size: int = 50, batch_size: int = 16, resolution: int = 10, span: float = 1.0) -> None
:canonical: src.utils.expo.landscape.plot_loss_landscape

```{autodoc2-docstring} src.utils.expo.landscape.plot_loss_landscape
```
````
