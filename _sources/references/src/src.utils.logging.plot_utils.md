# {py:mod}`src.utils.logging.plot_utils`

```{py:module} src.utils.logging.plot_utils
```

```{autodoc2-docstring} src.utils.logging.plot_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`draw_graph <src.utils.logging.plot_utils.draw_graph>`
  - ```{autodoc2-docstring} src.utils.logging.plot_utils.draw_graph
    :summary:
    ```
* - {py:obj}`plot_linechart <src.utils.logging.plot_utils.plot_linechart>`
  - ```{autodoc2-docstring} src.utils.logging.plot_utils.plot_linechart
    :summary:
    ```
* - {py:obj}`plot_tsp <src.utils.logging.plot_utils.plot_tsp>`
  - ```{autodoc2-docstring} src.utils.logging.plot_utils.plot_tsp
    :summary:
    ```
* - {py:obj}`discrete_cmap <src.utils.logging.plot_utils.discrete_cmap>`
  - ```{autodoc2-docstring} src.utils.logging.plot_utils.discrete_cmap
    :summary:
    ```
* - {py:obj}`plot_vehicle_routes <src.utils.logging.plot_utils.plot_vehicle_routes>`
  - ```{autodoc2-docstring} src.utils.logging.plot_utils.plot_vehicle_routes
    :summary:
    ```
* - {py:obj}`plot_attention_maps_wrapper <src.utils.logging.plot_utils.plot_attention_maps_wrapper>`
  - ```{autodoc2-docstring} src.utils.logging.plot_utils.plot_attention_maps_wrapper
    :summary:
    ```
* - {py:obj}`visualize_interactive_plot <src.utils.logging.plot_utils.visualize_interactive_plot>`
  - ```{autodoc2-docstring} src.utils.logging.plot_utils.visualize_interactive_plot
    :summary:
    ```
````

### API

````{py:function} draw_graph(distance_matrix)
:canonical: src.utils.logging.plot_utils.draw_graph

```{autodoc2-docstring} src.utils.logging.plot_utils.draw_graph
```
````

````{py:function} plot_linechart(output_dest, graph_log, plot_func, policies, x_label=None, y_label=None, title=None, fsave=True, scale='linear', x_values=None, linestyles=None, markers=None, annotate=True, pareto_front=False)
:canonical: src.utils.logging.plot_utils.plot_linechart

```{autodoc2-docstring} src.utils.logging.plot_utils.plot_linechart
```
````

````{py:function} plot_tsp(xy, tour, ax1)
:canonical: src.utils.logging.plot_utils.plot_tsp

```{autodoc2-docstring} src.utils.logging.plot_utils.plot_tsp
```
````

````{py:function} discrete_cmap(N, base_cmap=None)
:canonical: src.utils.logging.plot_utils.discrete_cmap

```{autodoc2-docstring} src.utils.logging.plot_utils.discrete_cmap
```
````

````{py:function} plot_vehicle_routes(data, route, ax1, markersize=5, visualize_demands=False, demand_scale=1, round_demand=False)
:canonical: src.utils.logging.plot_utils.plot_vehicle_routes

```{autodoc2-docstring} src.utils.logging.plot_utils.plot_vehicle_routes
```
````

````{py:function} plot_attention_maps_wrapper(dir_path, attention_dict, model_name, execution_function, layer_idx=0, sample_idx=0, head_idx=0, batch_idx=0, x_labels=None, y_labels=None, **execution_kwargs)
:canonical: src.utils.logging.plot_utils.plot_attention_maps_wrapper

```{autodoc2-docstring} src.utils.logging.plot_utils.plot_attention_maps_wrapper
```
````

````{py:function} visualize_interactive_plot(**kwargs)
:canonical: src.utils.logging.plot_utils.visualize_interactive_plot

```{autodoc2-docstring} src.utils.logging.plot_utils.visualize_interactive_plot
```
````
