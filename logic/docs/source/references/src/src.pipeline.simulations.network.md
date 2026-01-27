# {py:mod}`src.pipeline.simulations.network`

```{py:module} src.pipeline.simulations.network
```

```{autodoc2-docstring} src.pipeline.simulations.network
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DistanceStrategy <src.pipeline.simulations.network.DistanceStrategy>`
  - ```{autodoc2-docstring} src.pipeline.simulations.network.DistanceStrategy
    :summary:
    ```
* - {py:obj}`GoogleMapsStrategy <src.pipeline.simulations.network.GoogleMapsStrategy>`
  - ```{autodoc2-docstring} src.pipeline.simulations.network.GoogleMapsStrategy
    :summary:
    ```
* - {py:obj}`GeoPandasStrategy <src.pipeline.simulations.network.GeoPandasStrategy>`
  - ```{autodoc2-docstring} src.pipeline.simulations.network.GeoPandasStrategy
    :summary:
    ```
* - {py:obj}`OSMStrategy <src.pipeline.simulations.network.OSMStrategy>`
  - ```{autodoc2-docstring} src.pipeline.simulations.network.OSMStrategy
    :summary:
    ```
* - {py:obj}`IterativeDistanceStrategy <src.pipeline.simulations.network.IterativeDistanceStrategy>`
  - ```{autodoc2-docstring} src.pipeline.simulations.network.IterativeDistanceStrategy
    :summary:
    ```
* - {py:obj}`GeodesicStrategy <src.pipeline.simulations.network.GeodesicStrategy>`
  - ```{autodoc2-docstring} src.pipeline.simulations.network.GeodesicStrategy
    :summary:
    ```
* - {py:obj}`HaversineStrategy <src.pipeline.simulations.network.HaversineStrategy>`
  - ```{autodoc2-docstring} src.pipeline.simulations.network.HaversineStrategy
    :summary:
    ```
* - {py:obj}`EuclideanStrategy <src.pipeline.simulations.network.EuclideanStrategy>`
  - ```{autodoc2-docstring} src.pipeline.simulations.network.EuclideanStrategy
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`haversine_distance <src.pipeline.simulations.network.haversine_distance>`
  - ```{autodoc2-docstring} src.pipeline.simulations.network.haversine_distance
    :summary:
    ```
* - {py:obj}`compute_distance_matrix <src.pipeline.simulations.network.compute_distance_matrix>`
  - ```{autodoc2-docstring} src.pipeline.simulations.network.compute_distance_matrix
    :summary:
    ```
* - {py:obj}`apply_edges <src.pipeline.simulations.network.apply_edges>`
  - ```{autodoc2-docstring} src.pipeline.simulations.network.apply_edges
    :summary:
    ```
* - {py:obj}`get_paths_between_states <src.pipeline.simulations.network.get_paths_between_states>`
  - ```{autodoc2-docstring} src.pipeline.simulations.network.get_paths_between_states
    :summary:
    ```
````

### API

`````{py:class} DistanceStrategy
:canonical: src.pipeline.simulations.network.DistanceStrategy

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.simulations.network.DistanceStrategy
```

````{py:method} calculate(coords: pandas.DataFrame, **kwargs: typing.Any) -> numpy.ndarray
:canonical: src.pipeline.simulations.network.DistanceStrategy.calculate
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.network.DistanceStrategy.calculate
```

````

````{py:method} _eval_kwarg(kwarg: str, kwargs: typing.Dict[str, typing.Any]) -> bool
:canonical: src.pipeline.simulations.network.DistanceStrategy._eval_kwarg

```{autodoc2-docstring} src.pipeline.simulations.network.DistanceStrategy._eval_kwarg
```

````

`````

`````{py:class} GoogleMapsStrategy
:canonical: src.pipeline.simulations.network.GoogleMapsStrategy

Bases: {py:obj}`src.pipeline.simulations.network.DistanceStrategy`

```{autodoc2-docstring} src.pipeline.simulations.network.GoogleMapsStrategy
```

````{py:method} calculate(coords: pandas.DataFrame, **kwargs: typing.Any) -> numpy.ndarray
:canonical: src.pipeline.simulations.network.GoogleMapsStrategy.calculate

```{autodoc2-docstring} src.pipeline.simulations.network.GoogleMapsStrategy.calculate
```

````

`````

`````{py:class} GeoPandasStrategy
:canonical: src.pipeline.simulations.network.GeoPandasStrategy

Bases: {py:obj}`src.pipeline.simulations.network.DistanceStrategy`

```{autodoc2-docstring} src.pipeline.simulations.network.GeoPandasStrategy
```

````{py:method} calculate(coords: pandas.DataFrame, **kwargs: typing.Any) -> numpy.ndarray
:canonical: src.pipeline.simulations.network.GeoPandasStrategy.calculate

```{autodoc2-docstring} src.pipeline.simulations.network.GeoPandasStrategy.calculate
```

````

`````

`````{py:class} OSMStrategy
:canonical: src.pipeline.simulations.network.OSMStrategy

Bases: {py:obj}`src.pipeline.simulations.network.DistanceStrategy`

```{autodoc2-docstring} src.pipeline.simulations.network.OSMStrategy
```

````{py:method} calculate(coords: pandas.DataFrame, **kwargs: typing.Any) -> numpy.ndarray
:canonical: src.pipeline.simulations.network.OSMStrategy.calculate

```{autodoc2-docstring} src.pipeline.simulations.network.OSMStrategy.calculate
```

````

`````

`````{py:class} IterativeDistanceStrategy
:canonical: src.pipeline.simulations.network.IterativeDistanceStrategy

Bases: {py:obj}`src.pipeline.simulations.network.DistanceStrategy`

```{autodoc2-docstring} src.pipeline.simulations.network.IterativeDistanceStrategy
```

````{py:method} calculate_pair(coords_i: typing.Tuple[float, float], coords_j: typing.Tuple[float, float]) -> float
:canonical: src.pipeline.simulations.network.IterativeDistanceStrategy.calculate_pair
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.network.IterativeDistanceStrategy.calculate_pair
```

````

````{py:method} calculate(coords, **kwargs) -> numpy.ndarray
:canonical: src.pipeline.simulations.network.IterativeDistanceStrategy.calculate

```{autodoc2-docstring} src.pipeline.simulations.network.IterativeDistanceStrategy.calculate
```

````

`````

`````{py:class} GeodesicStrategy
:canonical: src.pipeline.simulations.network.GeodesicStrategy

Bases: {py:obj}`src.pipeline.simulations.network.IterativeDistanceStrategy`

```{autodoc2-docstring} src.pipeline.simulations.network.GeodesicStrategy
```

````{py:method} calculate_pair(coords_i: typing.Tuple[float, float], coords_j: typing.Tuple[float, float]) -> float
:canonical: src.pipeline.simulations.network.GeodesicStrategy.calculate_pair

```{autodoc2-docstring} src.pipeline.simulations.network.GeodesicStrategy.calculate_pair
```

````

`````

`````{py:class} HaversineStrategy
:canonical: src.pipeline.simulations.network.HaversineStrategy

Bases: {py:obj}`src.pipeline.simulations.network.IterativeDistanceStrategy`

```{autodoc2-docstring} src.pipeline.simulations.network.HaversineStrategy
```

````{py:method} calculate_pair(coords_i: typing.Tuple[float, float], coords_j: typing.Tuple[float, float]) -> float
:canonical: src.pipeline.simulations.network.HaversineStrategy.calculate_pair

```{autodoc2-docstring} src.pipeline.simulations.network.HaversineStrategy.calculate_pair
```

````

`````

`````{py:class} EuclideanStrategy
:canonical: src.pipeline.simulations.network.EuclideanStrategy

Bases: {py:obj}`src.pipeline.simulations.network.IterativeDistanceStrategy`

```{autodoc2-docstring} src.pipeline.simulations.network.EuclideanStrategy
```

````{py:method} calculate_pair(coords_i: typing.Tuple[float, float], coords_j: typing.Tuple[float, float]) -> float
:canonical: src.pipeline.simulations.network.EuclideanStrategy.calculate_pair

```{autodoc2-docstring} src.pipeline.simulations.network.EuclideanStrategy.calculate_pair
```

````

`````

````{py:function} haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float
:canonical: src.pipeline.simulations.network.haversine_distance

```{autodoc2-docstring} src.pipeline.simulations.network.haversine_distance
```
````

````{py:function} compute_distance_matrix(coords: pandas.DataFrame, method: str, **kwargs: typing.Any) -> numpy.ndarray
:canonical: src.pipeline.simulations.network.compute_distance_matrix

```{autodoc2-docstring} src.pipeline.simulations.network.compute_distance_matrix
```
````

````{py:function} apply_edges(dist_matrix: numpy.ndarray, edge_thresh: float, edge_method: typing.Optional[str]) -> typing.Tuple[numpy.ndarray, typing.Optional[typing.Dict[typing.Tuple[int, int], typing.List[int]]], typing.Optional[numpy.ndarray]]
:canonical: src.pipeline.simulations.network.apply_edges

```{autodoc2-docstring} src.pipeline.simulations.network.apply_edges
```
````

````{py:function} get_paths_between_states(n_bins: int, shortest_paths: typing.Optional[typing.Dict[typing.Tuple[int, int], typing.List[int]]] = None) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.pipeline.simulations.network.get_paths_between_states

```{autodoc2-docstring} src.pipeline.simulations.network.get_paths_between_states
```
````
