# {py:mod}`src.pipeline.ui.styles.kpi`

```{py:module} src.pipeline.ui.styles.kpi
```

```{autodoc2-docstring} src.pipeline.ui.styles.kpi
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`format_number <src.pipeline.ui.styles.kpi.format_number>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.kpi.format_number
    :summary:
    ```
* - {py:obj}`format_percentage <src.pipeline.ui.styles.kpi.format_percentage>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.kpi.format_percentage
    :summary:
    ```
* - {py:obj}`_format_delta <src.pipeline.ui.styles.kpi._format_delta>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.kpi._format_delta
    :summary:
    ```
* - {py:obj}`_delta_css_class <src.pipeline.ui.styles.kpi._delta_css_class>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.kpi._delta_css_class
    :summary:
    ```
* - {py:obj}`create_kpi_html <src.pipeline.ui.styles.kpi.create_kpi_html>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.kpi.create_kpi_html
    :summary:
    ```
* - {py:obj}`create_kpi_row <src.pipeline.ui.styles.kpi.create_kpi_row>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.kpi.create_kpi_row
    :summary:
    ```
* - {py:obj}`create_kpi_row_with_deltas <src.pipeline.ui.styles.kpi.create_kpi_row_with_deltas>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.kpi.create_kpi_row_with_deltas
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KPIValue <src.pipeline.ui.styles.kpi.KPIValue>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.kpi.KPIValue
    :summary:
    ```
* - {py:obj}`KPIDelta <src.pipeline.ui.styles.kpi.KPIDelta>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.kpi.KPIDelta
    :summary:
    ```
````

### API

````{py:data} KPIValue
:canonical: src.pipeline.ui.styles.kpi.KPIValue
:value: >
   None

```{autodoc2-docstring} src.pipeline.ui.styles.kpi.KPIValue
```

````

````{py:data} KPIDelta
:canonical: src.pipeline.ui.styles.kpi.KPIDelta
:value: >
   None

```{autodoc2-docstring} src.pipeline.ui.styles.kpi.KPIDelta
```

````

````{py:function} format_number(value: float, precision: int = 2) -> str
:canonical: src.pipeline.ui.styles.kpi.format_number

```{autodoc2-docstring} src.pipeline.ui.styles.kpi.format_number
```
````

````{py:function} format_percentage(value: float) -> str
:canonical: src.pipeline.ui.styles.kpi.format_percentage

```{autodoc2-docstring} src.pipeline.ui.styles.kpi.format_percentage
```
````

````{py:function} _format_delta(delta: float) -> str
:canonical: src.pipeline.ui.styles.kpi._format_delta

```{autodoc2-docstring} src.pipeline.ui.styles.kpi._format_delta
```
````

````{py:function} _delta_css_class(delta: float) -> str
:canonical: src.pipeline.ui.styles.kpi._delta_css_class

```{autodoc2-docstring} src.pipeline.ui.styles.kpi._delta_css_class
```
````

````{py:function} create_kpi_html(label: str, value: str, color: str = '#667eea', color_end: str = '#5a67d8', delta: typing.Optional[str] = None, delta_class: str = 'neutral', sparkline_svg: str = '') -> str
:canonical: src.pipeline.ui.styles.kpi.create_kpi_html

```{autodoc2-docstring} src.pipeline.ui.styles.kpi.create_kpi_html
```
````

````{py:function} create_kpi_row(metrics: dict, prefix: str = '') -> str
:canonical: src.pipeline.ui.styles.kpi.create_kpi_row

```{autodoc2-docstring} src.pipeline.ui.styles.kpi.create_kpi_row
```
````

````{py:function} create_kpi_row_with_deltas(metrics: typing.Dict[str, typing.Tuple[src.pipeline.ui.styles.kpi.KPIValue, src.pipeline.ui.styles.kpi.KPIDelta]], sparklines: typing.Optional[typing.Dict[str, str]] = None) -> str
:canonical: src.pipeline.ui.styles.kpi.create_kpi_row_with_deltas

```{autodoc2-docstring} src.pipeline.ui.styles.kpi.create_kpi_row_with_deltas
```
````
