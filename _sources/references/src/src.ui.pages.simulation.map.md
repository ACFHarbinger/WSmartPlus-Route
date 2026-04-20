# {py:mod}`src.ui.pages.simulation.map`

```{py:module} src.ui.pages.simulation.map
```

```{autodoc2-docstring} src.ui.pages.simulation.map
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`load_custom_matrix <src.ui.pages.simulation.map.load_custom_matrix>`
  - ```{autodoc2-docstring} src.ui.pages.simulation.map.load_custom_matrix
    :summary:
    ```
* - {py:obj}`reconstruct_tour <src.ui.pages.simulation.map.reconstruct_tour>`
  - ```{autodoc2-docstring} src.ui.pages.simulation.map.reconstruct_tour
    :summary:
    ```
* - {py:obj}`render_map_view <src.ui.pages.simulation.map.render_map_view>`
  - ```{autodoc2-docstring} src.ui.pages.simulation.map.render_map_view
    :summary:
    ```
* - {py:obj}`render_bin_heatmap <src.ui.pages.simulation.map.render_bin_heatmap>`
  - ```{autodoc2-docstring} src.ui.pages.simulation.map.render_bin_heatmap
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`template_dir <src.ui.pages.simulation.map.template_dir>`
  - ```{autodoc2-docstring} src.ui.pages.simulation.map.template_dir
    :summary:
    ```
* - {py:obj}`jinja_env <src.ui.pages.simulation.map.jinja_env>`
  - ```{autodoc2-docstring} src.ui.pages.simulation.map.jinja_env
    :summary:
    ```
````

### API

````{py:data} template_dir
:canonical: src.ui.pages.simulation.map.template_dir
:value: >
   'join(...)'

```{autodoc2-docstring} src.ui.pages.simulation.map.template_dir
```

````

````{py:data} jinja_env
:canonical: src.ui.pages.simulation.map.jinja_env
:value: >
   'Environment(...)'

```{autodoc2-docstring} src.ui.pages.simulation.map.jinja_env
```

````

````{py:function} load_custom_matrix(controls: typing.Dict[str, typing.Any]) -> typing.Any
:canonical: src.ui.pages.simulation.map.load_custom_matrix

```{autodoc2-docstring} src.ui.pages.simulation.map.load_custom_matrix
```
````

````{py:function} reconstruct_tour(tour: typing.List[typing.Any], all_bin_coords: typing.Optional[typing.List[typing.Dict[str, typing.Any]]]) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.ui.pages.simulation.map.reconstruct_tour

```{autodoc2-docstring} src.ui.pages.simulation.map.reconstruct_tour
```
````

````{py:function} render_map_view(display_entry: typing.Any, controls: typing.Dict[str, typing.Any]) -> None
:canonical: src.ui.pages.simulation.map.render_map_view

```{autodoc2-docstring} src.ui.pages.simulation.map.render_map_view
```
````

````{py:function} render_bin_heatmap(display_entry: typing.Any) -> None
:canonical: src.ui.pages.simulation.map.render_bin_heatmap

```{autodoc2-docstring} src.ui.pages.simulation.map.render_bin_heatmap
```
````
