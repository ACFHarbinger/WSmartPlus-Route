# {py:mod}`src.utils.expo`

```{py:module} src.utils.expo
```

```{autodoc2-docstring} src.utils.expo
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.utils.expo.heatmaps
src.utils.expo.helpers
src.utils.expo.charts
src.utils.expo.routes
src.utils.expo.landscape
src.utils.expo.embeddings
src.utils.expo.log_visualization
src.utils.expo.charts3d
src.utils.expo.interactive
src.utils.expo.attention
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`visualize_epoch <src.utils.expo.visualize_epoch>`
  - ```{autodoc2-docstring} src.utils.expo.visualize_epoch
    :summary:
    ```
* - {py:obj}`main <src.utils.expo.main>`
  - ```{autodoc2-docstring} src.utils.expo.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.utils.expo.__all__>`
  - ```{autodoc2-docstring} src.utils.expo.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: src.utils.expo.__all__
:value: >
   ['plot_linechart', 'plot_3dchart', 'draw_graph', 'plot_tsp', 'plot_vehicle_routes', 'discrete_cmap',...

```{autodoc2-docstring} src.utils.expo.__all__
```

````

````{py:function} visualize_epoch(model: typing.Any, problem: typing.Any, cfg: typing.Union[logic.src.configs.Config, omegaconf.DictConfig], epoch: int, tb_logger: typing.Any = None) -> None
:canonical: src.utils.expo.visualize_epoch

```{autodoc2-docstring} src.utils.expo.visualize_epoch
```
````

````{py:function} main()
:canonical: src.utils.expo.main

```{autodoc2-docstring} src.utils.expo.main
```
````
