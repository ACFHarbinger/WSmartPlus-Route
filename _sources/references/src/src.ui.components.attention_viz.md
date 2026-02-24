# {py:mod}`src.ui.components.attention_viz`

```{py:module} src.ui.components.attention_viz
```

```{autodoc2-docstring} src.ui.components.attention_viz
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`render_attention_viz <src.ui.components.attention_viz.render_attention_viz>`
  - ```{autodoc2-docstring} src.ui.components.attention_viz.render_attention_viz
    :summary:
    ```
* - {py:obj}`_render_geo <src.ui.components.attention_viz._render_geo>`
  - ```{autodoc2-docstring} src.ui.components.attention_viz._render_geo
    :summary:
    ```
* - {py:obj}`_render_bipartite <src.ui.components.attention_viz._render_bipartite>`
  - ```{autodoc2-docstring} src.ui.components.attention_viz._render_bipartite
    :summary:
    ```
* - {py:obj}`_render_heatmap <src.ui.components.attention_viz._render_heatmap>`
  - ```{autodoc2-docstring} src.ui.components.attention_viz._render_heatmap
    :summary:
    ```
* - {py:obj}`_to_numpy <src.ui.components.attention_viz._to_numpy>`
  - ```{autodoc2-docstring} src.ui.components.attention_viz._to_numpy
    :summary:
    ```
* - {py:obj}`_extract_head <src.ui.components.attention_viz._extract_head>`
  - ```{autodoc2-docstring} src.ui.components.attention_viz._extract_head
    :summary:
    ```
* - {py:obj}`_top_k_edges <src.ui.components.attention_viz._top_k_edges>`
  - ```{autodoc2-docstring} src.ui.components.attention_viz._top_k_edges
    :summary:
    ```
````

### API

````{py:function} render_attention_viz(hook_data: typing.Dict[str, typing.List[typing.Any]], node_coords: typing.Optional[numpy.ndarray] = None, decoding_step: int = 0, head_idx: int = 0, map_center: typing.Optional[typing.Tuple[float, float]] = None, map_zoom: int = 12, height: int = 500, min_edge_alpha: float = 0.05, top_k_edges: int = 20, color_scale: str = 'Reds', title: str = 'Attention Map') -> None
:canonical: src.ui.components.attention_viz.render_attention_viz

```{autodoc2-docstring} src.ui.components.attention_viz.render_attention_viz
```
````

````{py:function} _render_geo(attn: numpy.ndarray, coords: numpy.ndarray, center: typing.Optional[typing.Tuple[float, float]], zoom: int, top_k: int, min_alpha: float, height: int) -> typing.Any
:canonical: src.ui.components.attention_viz._render_geo

```{autodoc2-docstring} src.ui.components.attention_viz._render_geo
```
````

````{py:function} _render_bipartite(attn: numpy.ndarray, coords: numpy.ndarray, top_k: int, min_alpha: float, height: int, color_scale: str) -> typing.Any
:canonical: src.ui.components.attention_viz._render_bipartite

```{autodoc2-docstring} src.ui.components.attention_viz._render_bipartite
```
````

````{py:function} _render_heatmap(attn: numpy.ndarray, height: int, color_scale: str) -> typing.Any
:canonical: src.ui.components.attention_viz._render_heatmap

```{autodoc2-docstring} src.ui.components.attention_viz._render_heatmap
```
````

````{py:function} _to_numpy(tensor: typing.Any) -> numpy.ndarray
:canonical: src.ui.components.attention_viz._to_numpy

```{autodoc2-docstring} src.ui.components.attention_viz._to_numpy
```
````

````{py:function} _extract_head(attn_np: numpy.ndarray, head_idx: int = 0) -> numpy.ndarray
:canonical: src.ui.components.attention_viz._extract_head

```{autodoc2-docstring} src.ui.components.attention_viz._extract_head
```
````

````{py:function} _top_k_edges(attn: numpy.ndarray, top_k: int, min_alpha: float) -> typing.List[typing.Tuple[float, int, int]]
:canonical: src.ui.components.attention_viz._top_k_edges

```{autodoc2-docstring} src.ui.components.attention_viz._top_k_edges
```
````
