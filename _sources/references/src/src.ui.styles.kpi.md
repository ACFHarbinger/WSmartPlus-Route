# {py:mod}`src.ui.styles.kpi`

```{py:module} src.ui.styles.kpi
```

```{autodoc2-docstring} src.ui.styles.kpi
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KPIRenderer <src.ui.styles.kpi.KPIRenderer>`
  - ```{autodoc2-docstring} src.ui.styles.kpi.KPIRenderer
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`format_number <src.ui.styles.kpi.format_number>`
  - ```{autodoc2-docstring} src.ui.styles.kpi.format_number
    :summary:
    ```
* - {py:obj}`format_percentage <src.ui.styles.kpi.format_percentage>`
  - ```{autodoc2-docstring} src.ui.styles.kpi.format_percentage
    :summary:
    ```
* - {py:obj}`_format_delta <src.ui.styles.kpi._format_delta>`
  - ```{autodoc2-docstring} src.ui.styles.kpi._format_delta
    :summary:
    ```
* - {py:obj}`_delta_css_class <src.ui.styles.kpi._delta_css_class>`
  - ```{autodoc2-docstring} src.ui.styles.kpi._delta_css_class
    :summary:
    ```
* - {py:obj}`create_kpi_html <src.ui.styles.kpi.create_kpi_html>`
  - ```{autodoc2-docstring} src.ui.styles.kpi.create_kpi_html
    :summary:
    ```
* - {py:obj}`create_kpi_row <src.ui.styles.kpi.create_kpi_row>`
  - ```{autodoc2-docstring} src.ui.styles.kpi.create_kpi_row
    :summary:
    ```
* - {py:obj}`create_kpi_row_with_deltas <src.ui.styles.kpi.create_kpi_row_with_deltas>`
  - ```{autodoc2-docstring} src.ui.styles.kpi.create_kpi_row_with_deltas
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KPIValue <src.ui.styles.kpi.KPIValue>`
  - ```{autodoc2-docstring} src.ui.styles.kpi.KPIValue
    :summary:
    ```
* - {py:obj}`KPIDelta <src.ui.styles.kpi.KPIDelta>`
  - ```{autodoc2-docstring} src.ui.styles.kpi.KPIDelta
    :summary:
    ```
* - {py:obj}`renderer <src.ui.styles.kpi.renderer>`
  - ```{autodoc2-docstring} src.ui.styles.kpi.renderer
    :summary:
    ```
````

### API

````{py:data} KPIValue
:canonical: src.ui.styles.kpi.KPIValue
:value: >
   None

```{autodoc2-docstring} src.ui.styles.kpi.KPIValue
```

````

````{py:data} KPIDelta
:canonical: src.ui.styles.kpi.KPIDelta
:value: >
   None

```{autodoc2-docstring} src.ui.styles.kpi.KPIDelta
```

````

````{py:function} format_number(value: float, precision: int = 2) -> str
:canonical: src.ui.styles.kpi.format_number

```{autodoc2-docstring} src.ui.styles.kpi.format_number
```
````

````{py:function} format_percentage(value: float) -> str
:canonical: src.ui.styles.kpi.format_percentage

```{autodoc2-docstring} src.ui.styles.kpi.format_percentage
```
````

````{py:function} _format_delta(delta: float) -> str
:canonical: src.ui.styles.kpi._format_delta

```{autodoc2-docstring} src.ui.styles.kpi._format_delta
```
````

````{py:function} _delta_css_class(delta: float) -> str
:canonical: src.ui.styles.kpi._delta_css_class

```{autodoc2-docstring} src.ui.styles.kpi._delta_css_class
```
````

`````{py:class} KPIRenderer()
:canonical: src.ui.styles.kpi.KPIRenderer

```{autodoc2-docstring} src.ui.styles.kpi.KPIRenderer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.ui.styles.kpi.KPIRenderer.__init__
```

````{py:method} render_card(label: str, value: str, **kwargs) -> str
:canonical: src.ui.styles.kpi.KPIRenderer.render_card

```{autodoc2-docstring} src.ui.styles.kpi.KPIRenderer.render_card
```

````

````{py:method} render_row(cards_html: typing.List[str]) -> str
:canonical: src.ui.styles.kpi.KPIRenderer.render_row

```{autodoc2-docstring} src.ui.styles.kpi.KPIRenderer.render_row
```

````

`````

````{py:data} renderer
:canonical: src.ui.styles.kpi.renderer
:value: >
   'KPIRenderer(...)'

```{autodoc2-docstring} src.ui.styles.kpi.renderer
```

````

````{py:function} create_kpi_html(label: str, value: str, color: str = '#667eea', color_end: str = '#5a67d8', delta: typing.Optional[str] = None, delta_class: str = 'neutral', sparkline_svg: str = '') -> str
:canonical: src.ui.styles.kpi.create_kpi_html

```{autodoc2-docstring} src.ui.styles.kpi.create_kpi_html
```
````

````{py:function} create_kpi_row(metrics: typing.Dict[str, typing.Union[float, int, str]], prefix: str = '') -> str
:canonical: src.ui.styles.kpi.create_kpi_row

```{autodoc2-docstring} src.ui.styles.kpi.create_kpi_row
```
````

````{py:function} create_kpi_row_with_deltas(metrics: typing.Dict[str, typing.Tuple[src.ui.styles.kpi.KPIValue, src.ui.styles.kpi.KPIDelta]], sparklines: typing.Optional[typing.Dict[str, str]] = None) -> str
:canonical: src.ui.styles.kpi.create_kpi_row_with_deltas

```{autodoc2-docstring} src.ui.styles.kpi.create_kpi_row_with_deltas
```
````
