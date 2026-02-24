# {py:mod}`src.ui.components.policy_viz`

```{py:module} src.ui.components.policy_viz
```

```{autodoc2-docstring} src.ui.components.policy_viz
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`render_policy_viz <src.ui.components.policy_viz.render_policy_viz>`
  - ```{autodoc2-docstring} src.ui.components.policy_viz.render_policy_viz
    :summary:
    ```
* - {py:obj}`_ema <src.ui.components.policy_viz._ema>`
  - ```{autodoc2-docstring} src.ui.components.policy_viz._ema
    :summary:
    ```
* - {py:obj}`_bar_counts <src.ui.components.policy_viz._bar_counts>`
  - ```{autodoc2-docstring} src.ui.components.policy_viz._bar_counts
    :summary:
    ```
* - {py:obj}`_render_alns <src.ui.components.policy_viz._render_alns>`
  - ```{autodoc2-docstring} src.ui.components.policy_viz._render_alns
    :summary:
    ```
* - {py:obj}`_render_hgs <src.ui.components.policy_viz._render_hgs>`
  - ```{autodoc2-docstring} src.ui.components.policy_viz._render_hgs
    :summary:
    ```
* - {py:obj}`_render_aco <src.ui.components.policy_viz._render_aco>`
  - ```{autodoc2-docstring} src.ui.components.policy_viz._render_aco
    :summary:
    ```
* - {py:obj}`_render_ils <src.ui.components.policy_viz._render_ils>`
  - ```{autodoc2-docstring} src.ui.components.policy_viz._render_ils
    :summary:
    ```
* - {py:obj}`_render_selector <src.ui.components.policy_viz._render_selector>`
  - ```{autodoc2-docstring} src.ui.components.policy_viz._render_selector
    :summary:
    ```
* - {py:obj}`_render_rls <src.ui.components.policy_viz._render_rls>`
  - ```{autodoc2-docstring} src.ui.components.policy_viz._render_rls
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_ALNS_OP_NAMES <src.ui.components.policy_viz._ALNS_OP_NAMES>`
  - ```{autodoc2-docstring} src.ui.components.policy_viz._ALNS_OP_NAMES
    :summary:
    ```
* - {py:obj}`_PERTURB_COLORS <src.ui.components.policy_viz._PERTURB_COLORS>`
  - ```{autodoc2-docstring} src.ui.components.policy_viz._PERTURB_COLORS
    :summary:
    ```
````

### API

````{py:function} render_policy_viz(viz_data: typing.Dict[str, typing.List[typing.Any]], height: int = 400, title: typing.Optional[str] = None, smooth_window: int = 1) -> None
:canonical: src.ui.components.policy_viz.render_policy_viz

```{autodoc2-docstring} src.ui.components.policy_viz.render_policy_viz
```
````

````{py:function} _ema(values: typing.List[float], window: int) -> typing.List[float]
:canonical: src.ui.components.policy_viz._ema

```{autodoc2-docstring} src.ui.components.policy_viz._ema
```
````

````{py:function} _bar_counts(values: typing.List[typing.Any]) -> typing.Dict[typing.Any, int]
:canonical: src.ui.components.policy_viz._bar_counts

```{autodoc2-docstring} src.ui.components.policy_viz._bar_counts
```
````

````{py:data} _ALNS_OP_NAMES
:canonical: src.ui.components.policy_viz._ALNS_OP_NAMES
:value: >
   None

```{autodoc2-docstring} src.ui.components.policy_viz._ALNS_OP_NAMES
```

````

````{py:function} _render_alns(data: typing.Dict[str, typing.List[typing.Any]], height: int, smooth: int) -> None
:canonical: src.ui.components.policy_viz._render_alns

```{autodoc2-docstring} src.ui.components.policy_viz._render_alns
```
````

````{py:function} _render_hgs(data: typing.Dict[str, typing.List[typing.Any]], height: int, smooth: int) -> None
:canonical: src.ui.components.policy_viz._render_hgs

```{autodoc2-docstring} src.ui.components.policy_viz._render_hgs
```
````

````{py:function} _render_aco(data: typing.Dict[str, typing.List[typing.Any]], height: int, smooth: int) -> None
:canonical: src.ui.components.policy_viz._render_aco

```{autodoc2-docstring} src.ui.components.policy_viz._render_aco
```
````

````{py:data} _PERTURB_COLORS
:canonical: src.ui.components.policy_viz._PERTURB_COLORS
:value: >
   None

```{autodoc2-docstring} src.ui.components.policy_viz._PERTURB_COLORS
```

````

````{py:function} _render_ils(data: typing.Dict[str, typing.List[typing.Any]], height: int, smooth: int) -> None
:canonical: src.ui.components.policy_viz._render_ils

```{autodoc2-docstring} src.ui.components.policy_viz._render_ils
```
````

````{py:function} _render_selector(data: typing.Dict[str, typing.List[typing.Any]], height: int) -> None
:canonical: src.ui.components.policy_viz._render_selector

```{autodoc2-docstring} src.ui.components.policy_viz._render_selector
```
````

````{py:function} _render_rls(data: typing.Dict[str, typing.List[typing.Any]], height: int, smooth: int) -> None
:canonical: src.ui.components.policy_viz._render_rls

```{autodoc2-docstring} src.ui.components.policy_viz._render_rls
```
````
