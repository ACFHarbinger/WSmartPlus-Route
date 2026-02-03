# {py:mod}`src.utils.logging.visualization.landscape`

```{py:module} src.utils.logging.visualization.landscape
```

```{autodoc2-docstring} src.utils.logging.visualization.landscape
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`imitation_loss_fn <src.utils.logging.visualization.landscape.imitation_loss_fn>`
  - ```{autodoc2-docstring} src.utils.logging.visualization.landscape.imitation_loss_fn
    :summary:
    ```
* - {py:obj}`rl_loss_fn <src.utils.logging.visualization.landscape.rl_loss_fn>`
  - ```{autodoc2-docstring} src.utils.logging.visualization.landscape.rl_loss_fn
    :summary:
    ```
* - {py:obj}`plot_loss_landscape <src.utils.logging.visualization.landscape.plot_loss_landscape>`
  - ```{autodoc2-docstring} src.utils.logging.visualization.landscape.plot_loss_landscape
    :summary:
    ```
````

### API

````{py:function} imitation_loss_fn(m, x_batch, pi_target, cost_weights=None)
:canonical: src.utils.logging.visualization.landscape.imitation_loss_fn

```{autodoc2-docstring} src.utils.logging.visualization.landscape.imitation_loss_fn
```
````

````{py:function} rl_loss_fn(m, x_batch, cost_weights=None)
:canonical: src.utils.logging.visualization.landscape.rl_loss_fn

```{autodoc2-docstring} src.utils.logging.visualization.landscape.rl_loss_fn
```
````

````{py:function} plot_loss_landscape(model, opts, output_dir, epoch=0, size=50, batch_size=16, resolution=10, span=1.0)
:canonical: src.utils.logging.visualization.landscape.plot_loss_landscape

```{autodoc2-docstring} src.utils.logging.visualization.landscape.plot_loss_landscape
```
````
