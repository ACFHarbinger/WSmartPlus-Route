# {py:mod}`src.utils.routing.tours`

```{py:module} src.utils.routing.tours
```

```{autodoc2-docstring} src.utils.routing.tours
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_route_cost <src.utils.routing.tours.get_route_cost>`
  - ```{autodoc2-docstring} src.utils.routing.tours.get_route_cost
    :summary:
    ```
* - {py:obj}`_trip_time_matrix <src.utils.routing.tours._trip_time_matrix>`
  - ```{autodoc2-docstring} src.utils.routing.tours._trip_time_matrix
    :summary:
    ```
* - {py:obj}`get_multi_tour <src.utils.routing.tours.get_multi_tour>`
  - ```{autodoc2-docstring} src.utils.routing.tours.get_multi_tour
    :summary:
    ```
````

### API

````{py:function} get_route_cost(distancesC, tour)
:canonical: src.utils.routing.tours.get_route_cost

```{autodoc2-docstring} src.utils.routing.tours.get_route_cost
```
````

````{py:function} _trip_time_matrix(distance_matrix: numpy.ndarray, time_matrix: typing.Optional[numpy.ndarray], avg_speed_kmh: typing.Optional[float], shift_hours: float, service_time_h: float) -> numpy.ndarray
:canonical: src.utils.routing.tours._trip_time_matrix

```{autodoc2-docstring} src.utils.routing.tours._trip_time_matrix
```
````

````{py:function} get_multi_tour(tour: typing.List[int], bins_waste: numpy.ndarray, max_capacity: float, distance_matrix: numpy.ndarray, shift_hours: typing.Optional[float] = None, avg_speed_kmh: typing.Optional[float] = None, service_time_h: typing.Optional[float] = None, time_matrix: typing.Optional[numpy.ndarray] = None) -> typing.List[int]
:canonical: src.utils.routing.tours.get_multi_tour

```{autodoc2-docstring} src.utils.routing.tours.get_multi_tour
```
````
