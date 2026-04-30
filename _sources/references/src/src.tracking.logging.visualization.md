# {py:mod}`src.tracking.logging.visualization`

```{py:module} src.tracking.logging.visualization
```

```{autodoc2-docstring} src.tracking.logging.visualization
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.tracking.logging.visualization.heatmaps
src.tracking.logging.visualization.landscape
src.tracking.logging.visualization.helpers
src.tracking.logging.visualization.embeddings
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`visualize_epoch <src.tracking.logging.visualization.visualize_epoch>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.visualize_epoch
    :summary:
    ```
* - {py:obj}`main <src.tracking.logging.visualization.main>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.tracking.logging.visualization.__all__>`
  - ```{autodoc2-docstring} src.tracking.logging.visualization.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: src.tracking.logging.visualization.__all__
:value: >
   ['visualize_epoch', 'log_weight_distributions', 'plot_weight_trajectories', 'project_node_embeddings...

```{autodoc2-docstring} src.tracking.logging.visualization.__all__
```

````

````{py:function} visualize_epoch(model: typing.Any, problem: typing.Any, cfg: typing.Union[logic.src.configs.Config, omegaconf.DictConfig], epoch: int, tb_logger: typing.Any = None) -> None
:canonical: src.tracking.logging.visualization.visualize_epoch

```{autodoc2-docstring} src.tracking.logging.visualization.visualize_epoch
```
````

````{py:function} main()
:canonical: src.tracking.logging.visualization.main

```{autodoc2-docstring} src.tracking.logging.visualization.main
```
````
