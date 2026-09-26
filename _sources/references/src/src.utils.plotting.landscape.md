# {py:mod}`src.utils.plotting.landscape`

```{py:module} src.utils.plotting.landscape
```

```{autodoc2-docstring} src.utils.plotting.landscape
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImitationMetric <src.utils.plotting.landscape.ImitationMetric>`
  - ```{autodoc2-docstring} src.utils.plotting.landscape.ImitationMetric
    :summary:
    ```
* - {py:obj}`RLMetric <src.utils.plotting.landscape.RLMetric>`
  - ```{autodoc2-docstring} src.utils.plotting.landscape.RLMetric
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`imitation_loss_fn <src.utils.plotting.landscape.imitation_loss_fn>`
  - ```{autodoc2-docstring} src.utils.plotting.landscape.imitation_loss_fn
    :summary:
    ```
* - {py:obj}`rl_loss_fn <src.utils.plotting.landscape.rl_loss_fn>`
  - ```{autodoc2-docstring} src.utils.plotting.landscape.rl_loss_fn
    :summary:
    ```
* - {py:obj}`plot_loss_landscape <src.utils.plotting.landscape.plot_loss_landscape>`
  - ```{autodoc2-docstring} src.utils.plotting.landscape.plot_loss_landscape
    :summary:
    ```
````

### API

````{py:function} imitation_loss_fn(m, x_batch, pi_target, cost_weights=None)
:canonical: src.utils.plotting.landscape.imitation_loss_fn

```{autodoc2-docstring} src.utils.plotting.landscape.imitation_loss_fn
```
````

````{py:function} rl_loss_fn(m, x_batch, cost_weights=None)
:canonical: src.utils.plotting.landscape.rl_loss_fn

```{autodoc2-docstring} src.utils.plotting.landscape.rl_loss_fn
```
````

`````{py:class} ImitationMetric(x_batch, pi_target, cost_weights=None)
:canonical: src.utils.plotting.landscape.ImitationMetric

Bases: {py:obj}`loss_landscapes.metrics.Metric`

```{autodoc2-docstring} src.utils.plotting.landscape.ImitationMetric
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.plotting.landscape.ImitationMetric.__init__
```

````{py:method} __call__(model_wrapper)
:canonical: src.utils.plotting.landscape.ImitationMetric.__call__

```{autodoc2-docstring} src.utils.plotting.landscape.ImitationMetric.__call__
```

````

`````

`````{py:class} RLMetric(x_batch, cost_weights=None)
:canonical: src.utils.plotting.landscape.RLMetric

Bases: {py:obj}`loss_landscapes.metrics.Metric`

```{autodoc2-docstring} src.utils.plotting.landscape.RLMetric
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.plotting.landscape.RLMetric.__init__
```

````{py:method} __call__(model_wrapper)
:canonical: src.utils.plotting.landscape.RLMetric.__call__

```{autodoc2-docstring} src.utils.plotting.landscape.RLMetric.__call__
```

````

`````

````{py:function} plot_loss_landscape(model: typing.Any, cfg: typing.Union[logic.src.configs.Config, omegaconf.DictConfig], output_dir: str, epoch: int = 0, size: int = 50, batch_size: int = 16, resolution: int = 10, span: float = 1.0) -> None
:canonical: src.utils.plotting.landscape.plot_loss_landscape

```{autodoc2-docstring} src.utils.plotting.landscape.plot_loss_landscape
```
````
