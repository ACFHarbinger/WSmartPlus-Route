# {py:mod}`src.utils.logging.modules.gui`

```{py:module} src.utils.logging.modules.gui
```

```{autodoc2-docstring} src.utils.logging.modules.gui
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`send_daily_output_to_gui <src.utils.logging.modules.gui.send_daily_output_to_gui>`
  - ```{autodoc2-docstring} src.utils.logging.modules.gui.send_daily_output_to_gui
    :summary:
    ```
* - {py:obj}`send_final_output_to_gui <src.utils.logging.modules.gui.send_final_output_to_gui>`
  - ```{autodoc2-docstring} src.utils.logging.modules.gui.send_final_output_to_gui
    :summary:
    ```
* - {py:obj}`_get_lat_lon <src.utils.logging.modules.gui._get_lat_lon>`
  - ```{autodoc2-docstring} src.utils.logging.modules.gui._get_lat_lon
    :summary:
    ```
* - {py:obj}`_process_tour_point <src.utils.logging.modules.gui._process_tour_point>`
  - ```{autodoc2-docstring} src.utils.logging.modules.gui._process_tour_point
    :summary:
    ```
* - {py:obj}`_build_all_bin_coords <src.utils.logging.modules.gui._build_all_bin_coords>`
  - ```{autodoc2-docstring} src.utils.logging.modules.gui._build_all_bin_coords
    :summary:
    ```
````

### API

````{py:function} send_daily_output_to_gui(daily_log: typing.Dict[str, typing.Any], policy: str, sample_idx: int, day: int, bins_c: typing.Sequence[float], collected: typing.Sequence[float], bins_real_c_after: typing.Sequence[float], log_path: str, tour: typing.Sequence[int], coordinates: typing.Union[pandas.DataFrame, typing.List[typing.Any]], lock: typing.Optional[threading.Lock] = None, must_go: typing.Optional[typing.Sequence[int]] = None) -> None
:canonical: src.utils.logging.modules.gui.send_daily_output_to_gui

```{autodoc2-docstring} src.utils.logging.modules.gui.send_daily_output_to_gui
```
````

````{py:function} send_final_output_to_gui(log: typing.Dict[str, typing.Any], log_std: typing.Optional[typing.Dict[str, typing.Any]], n_samples: int, policies: typing.List[str], log_path: str, lock: typing.Optional[threading.Lock] = None) -> None
:canonical: src.utils.logging.modules.gui.send_final_output_to_gui

```{autodoc2-docstring} src.utils.logging.modules.gui.send_final_output_to_gui
```
````

````{py:function} _get_lat_lon(row: pandas.Series) -> tuple
:canonical: src.utils.logging.modules.gui._get_lat_lon

```{autodoc2-docstring} src.utils.logging.modules.gui._get_lat_lon
```
````

````{py:function} _process_tour_point(node_idx: int, coords_lookup: typing.Optional[pandas.DataFrame]) -> typing.Dict[str, typing.Any]
:canonical: src.utils.logging.modules.gui._process_tour_point

```{autodoc2-docstring} src.utils.logging.modules.gui._process_tour_point
```
````

````{py:function} _build_all_bin_coords(coords_lookup: pandas.DataFrame, n_bins: int) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.utils.logging.modules.gui._build_all_bin_coords

```{autodoc2-docstring} src.utils.logging.modules.gui._build_all_bin_coords
```
````
