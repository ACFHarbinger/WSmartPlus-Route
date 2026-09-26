# {py:mod}`src.utils.plotting`

```{py:module} src.utils.plotting
```

```{autodoc2-docstring} src.utils.plotting
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.utils.plotting.log_visualization
src.utils.plotting.heatmaps
src.utils.plotting.embeddings
src.utils.plotting.helpers
src.utils.plotting.interactive
src.utils.plotting.landscape
src.utils.plotting.attention
src.utils.plotting.charts3d
src.utils.plotting.charts
src.utils.plotting.routes
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`visualize_epoch <src.utils.plotting.visualize_epoch>`
  - ```{autodoc2-docstring} src.utils.plotting.visualize_epoch
    :summary:
    ```
* - {py:obj}`main <src.utils.plotting.main>`
  - ```{autodoc2-docstring} src.utils.plotting.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.utils.plotting.__all__>`
  - ```{autodoc2-docstring} src.utils.plotting.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: src.utils.plotting.__all__
:value: >
   ['plot_linechart', 'plot_3dchart', 'draw_graph', 'plot_tsp', 'plot_vehicle_routes', 'discrete_cmap',...

```{autodoc2-docstring} src.utils.plotting.__all__
```

````

````{py:function} visualize_epoch(model: typing.Any, problem: typing.Any, cfg: typing.Union[logic.src.configs.Config, omegaconf.DictConfig], epoch: int, tb_logger: typing.Any = None) -> None
:canonical: src.utils.plotting.visualize_epoch

```{autodoc2-docstring} src.utils.plotting.visualize_epoch
```
````

````{py:function} main()
:canonical: src.utils.plotting.main

```{autodoc2-docstring} src.utils.plotting.main
```
````
