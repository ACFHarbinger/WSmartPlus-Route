# {py:mod}`src.tracking.logging.visualization.landscape`

```{py:module} src.tracking.logging.visualization.landscape
```

```{autodoc2-docstring} src.tracking.logging.visualization.landscape
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImitationMetric <src.tracking.logging.visualization.landscape.ImitationMetric>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.landscape.ImitationMetric
    :summary:
    ```
* - {py:obj}`RLMetric <src.tracking.logging.visualization.landscape.RLMetric>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.landscape.RLMetric
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`imitation_loss_fn <src.tracking.logging.visualization.landscape.imitation_loss_fn>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.landscape.imitation_loss_fn
    :summary:
    ```
* - {py:obj}`rl_loss_fn <src.tracking.logging.visualization.landscape.rl_loss_fn>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.landscape.rl_loss_fn
    :summary:
    ```
* - {py:obj}`plot_loss_landscape <src.tracking.logging.visualization.landscape.plot_loss_landscape>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.landscape.plot_loss_landscape
    :summary:
    ```
````

### API

````{py:function} imitation_loss_fn(m, x_batch, pi_target, cost_weights=None)
:canonical: src.tracking.logging.visualization.landscape.imitation_loss_fn

```{autodoc2-docstring} src.tracking.logging.visualization.landscape.imitation_loss_fn
```
````

````{py:function} rl_loss_fn(m, x_batch, cost_weights=None)
:canonical: src.tracking.logging.visualization.landscape.rl_loss_fn

```{autodoc2-docstring} src.tracking.logging.visualization.landscape.rl_loss_fn
```
````

`````{py:class} ImitationMetric(x_batch, pi_target, cost_weights=None)
:canonical: src.tracking.logging.visualization.landscape.ImitationMetric

Bases: {py:obj}`loss_landscapes.metrics.Metric`

```{autodoc2-docstring} src.tracking.logging.visualization.landscape.ImitationMetric
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.logging.visualization.landscape.ImitationMetric.__init__
```

````{py:method} __call__(model_wrapper)
:canonical: src.tracking.logging.visualization.landscape.ImitationMetric.__call__

```{autodoc2-docstring} src.tracking.logging.visualization.landscape.ImitationMetric.__call__
```

````

`````

`````{py:class} RLMetric(x_batch, cost_weights=None)
:canonical: src.tracking.logging.visualization.landscape.RLMetric

Bases: {py:obj}`loss_landscapes.metrics.Metric`

```{autodoc2-docstring} src.tracking.logging.visualization.landscape.RLMetric
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.logging.visualization.landscape.RLMetric.__init__
```

````{py:method} __call__(model_wrapper)
:canonical: src.tracking.logging.visualization.landscape.RLMetric.__call__

```{autodoc2-docstring} src.tracking.logging.visualization.landscape.RLMetric.__call__
```

````

`````

````{py:function} plot_loss_landscape(model: typing.Any, cfg: typing.Union[logic.src.configs.Config, omegaconf.DictConfig], output_dir: str, epoch: int = 0, size: int = 50, batch_size: int = 16, resolution: int = 10, span: float = 1.0) -> None
:canonical: src.tracking.logging.visualization.landscape.plot_loss_landscape

```{autodoc2-docstring} src.tracking.logging.visualization.landscape.plot_loss_landscape
```
````
