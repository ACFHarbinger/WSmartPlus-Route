# {py:mod}`src.pipeline.ui.styles.styling`

```{py:module} src.pipeline.ui.styles.styling
```

```{autodoc2-docstring} src.pipeline.ui.styles.styling
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_page_config <src.pipeline.ui.styles.styling.get_page_config>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.styling.get_page_config
    :summary:
    ```
* - {py:obj}`format_number <src.pipeline.ui.styles.styling.format_number>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.styling.format_number
    :summary:
    ```
* - {py:obj}`format_percentage <src.pipeline.ui.styles.styling.format_percentage>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.styling.format_percentage
    :summary:
    ```
* - {py:obj}`create_kpi_html <src.pipeline.ui.styles.styling.create_kpi_html>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.styling.create_kpi_html
    :summary:
    ```
* - {py:obj}`create_kpi_row <src.pipeline.ui.styles.styling.create_kpi_row>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.styling.create_kpi_row
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CUSTOM_CSS <src.pipeline.ui.styles.styling.CUSTOM_CSS>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.styling.CUSTOM_CSS
    :summary:
    ```
* - {py:obj}`CHART_COLORS <src.pipeline.ui.styles.styling.CHART_COLORS>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.styling.CHART_COLORS
    :summary:
    ```
* - {py:obj}`ROUTE_COLORS <src.pipeline.ui.styles.styling.ROUTE_COLORS>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.styling.ROUTE_COLORS
    :summary:
    ```
* - {py:obj}`STATUS_COLORS <src.pipeline.ui.styles.styling.STATUS_COLORS>`
  - ```{autodoc2-docstring} src.pipeline.ui.styles.styling.STATUS_COLORS
    :summary:
    ```
````

### API

````{py:data} CUSTOM_CSS
:canonical: src.pipeline.ui.styles.styling.CUSTOM_CSS
:value: <Multiline-String>

```{autodoc2-docstring} src.pipeline.ui.styles.styling.CUSTOM_CSS
```

````

````{py:data} CHART_COLORS
:canonical: src.pipeline.ui.styles.styling.CHART_COLORS
:value: >
   ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', ...

```{autodoc2-docstring} src.pipeline.ui.styles.styling.CHART_COLORS
```

````

````{py:data} ROUTE_COLORS
:canonical: src.pipeline.ui.styles.styling.ROUTE_COLORS
:value: >
   ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

```{autodoc2-docstring} src.pipeline.ui.styles.styling.ROUTE_COLORS
```

````

````{py:data} STATUS_COLORS
:canonical: src.pipeline.ui.styles.styling.STATUS_COLORS
:value: >
   None

```{autodoc2-docstring} src.pipeline.ui.styles.styling.STATUS_COLORS
```

````

````{py:function} get_page_config() -> dict
:canonical: src.pipeline.ui.styles.styling.get_page_config

```{autodoc2-docstring} src.pipeline.ui.styles.styling.get_page_config
```
````

````{py:function} format_number(value: float, precision: int = 2) -> str
:canonical: src.pipeline.ui.styles.styling.format_number

```{autodoc2-docstring} src.pipeline.ui.styles.styling.format_number
```
````

````{py:function} format_percentage(value: float) -> str
:canonical: src.pipeline.ui.styles.styling.format_percentage

```{autodoc2-docstring} src.pipeline.ui.styles.styling.format_percentage
```
````

````{py:function} create_kpi_html(label: str, value: str, color: str = '#667eea') -> str
:canonical: src.pipeline.ui.styles.styling.create_kpi_html

```{autodoc2-docstring} src.pipeline.ui.styles.styling.create_kpi_html
```
````

````{py:function} create_kpi_row(metrics: dict) -> str
:canonical: src.pipeline.ui.styles.styling.create_kpi_row

```{autodoc2-docstring} src.pipeline.ui.styles.styling.create_kpi_row
```
````
