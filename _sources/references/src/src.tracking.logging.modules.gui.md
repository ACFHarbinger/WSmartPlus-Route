# {py:mod}`src.tracking.logging.modules.gui`

```{py:module} src.tracking.logging.modules.gui
```

```{autodoc2-docstring} src.tracking.logging.modules.gui
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`send_daily_output_to_gui <src.tracking.logging.modules.gui.send_daily_output_to_gui>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.gui.send_daily_output_to_gui
    :summary:
    ```
* - {py:obj}`send_final_output_to_gui <src.tracking.logging.modules.gui.send_final_output_to_gui>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.gui.send_final_output_to_gui
    :summary:
    ```
* - {py:obj}`_get_lat_lon <src.tracking.logging.modules.gui._get_lat_lon>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.gui._get_lat_lon
    :summary:
    ```
* - {py:obj}`_process_tour_point <src.tracking.logging.modules.gui._process_tour_point>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.gui._process_tour_point
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`template_dir <src.tracking.logging.modules.gui.template_dir>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.gui.template_dir
    :summary:
    ```
* - {py:obj}`jinja_env <src.tracking.logging.modules.gui.jinja_env>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.gui.jinja_env
    :summary:
    ```
* - {py:obj}`POPUP_TEMPLATE <src.tracking.logging.modules.gui.POPUP_TEMPLATE>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.gui.POPUP_TEMPLATE
    :summary:
    ```
````

### API

````{py:data} template_dir
:canonical: src.tracking.logging.modules.gui.template_dir
:value: >
   'dirname(...)'

```{autodoc2-docstring} src.tracking.logging.modules.gui.template_dir
```

````

````{py:data} jinja_env
:canonical: src.tracking.logging.modules.gui.jinja_env
:value: >
   'Environment(...)'

```{autodoc2-docstring} src.tracking.logging.modules.gui.jinja_env
```

````

````{py:data} POPUP_TEMPLATE
:canonical: src.tracking.logging.modules.gui.POPUP_TEMPLATE
:value: >
   'get_template(...)'

```{autodoc2-docstring} src.tracking.logging.modules.gui.POPUP_TEMPLATE
```

````

````{py:function} send_daily_output_to_gui(daily_log: typing.Dict[str, typing.Any], policy: str, sample_idx: int, day: int, bins_c: typing.Sequence[float], collected: typing.Sequence[float], bins_real_c_after: typing.Sequence[float], log_path: str, tour: typing.Sequence[int], coordinates: typing.Union[pandas.DataFrame, typing.List[typing.Any]], lock: typing.Optional[threading.Lock] = None, mandatory: typing.Optional[typing.Sequence[int]] = None) -> None
:canonical: src.tracking.logging.modules.gui.send_daily_output_to_gui

```{autodoc2-docstring} src.tracking.logging.modules.gui.send_daily_output_to_gui
```
````

````{py:function} send_final_output_to_gui(log: typing.Dict[str, typing.Any], log_std: typing.Optional[typing.Dict[str, typing.Any]], n_samples: int, policies: typing.List[str], log_path: str, lock: typing.Optional[threading.Lock] = None) -> None
:canonical: src.tracking.logging.modules.gui.send_final_output_to_gui

```{autodoc2-docstring} src.tracking.logging.modules.gui.send_final_output_to_gui
```
````

````{py:function} _get_lat_lon(row: pandas.Series) -> typing.Tuple[typing.Optional[float], typing.Optional[float]]
:canonical: src.tracking.logging.modules.gui._get_lat_lon

```{autodoc2-docstring} src.tracking.logging.modules.gui._get_lat_lon
```
````

````{py:function} _process_tour_point(node_idx: int, coords_lookup: typing.Optional[pandas.DataFrame]) -> typing.Dict[str, typing.Any]
:canonical: src.tracking.logging.modules.gui._process_tour_point

```{autodoc2-docstring} src.tracking.logging.modules.gui._process_tour_point
```
````
