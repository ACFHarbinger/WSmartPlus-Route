# {py:mod}`src.data.network`

```{py:module} src.data.network
```

```{autodoc2-docstring} src.data.network
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

src.data.network.base
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.data.network.osm
src.data.network.geodesic
src.data.network.haversine
src.data.network.geopandas
src.data.network.euclidean
src.data.network.google
src.data.network.file
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`haversine_distance <src.data.network.haversine_distance>`
  - ```{autodoc2-docstring} src.data.network.haversine_distance
    :summary:
    ```
* - {py:obj}`compute_distance_matrix <src.data.network.compute_distance_matrix>`
  - ```{autodoc2-docstring} src.data.network.compute_distance_matrix
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.data.network.__all__>`
  - ```{autodoc2-docstring} src.data.network.__all__
    :summary:
    ```
````

### API

````{py:function} haversine_distance(lat1: typing.Union[float, numpy.ndarray, pandas.Series], lng1: typing.Union[float, numpy.ndarray, pandas.Series], lat2: typing.Union[float, numpy.ndarray, pandas.Series], lng2: typing.Union[float, numpy.ndarray, pandas.Series]) -> typing.Union[float, numpy.ndarray]
:canonical: src.data.network.haversine_distance

```{autodoc2-docstring} src.data.network.haversine_distance
```
````

````{py:function} compute_distance_matrix(coords: pandas.DataFrame, method: str, **kwargs: typing.Any) -> numpy.ndarray
:canonical: src.data.network.compute_distance_matrix

```{autodoc2-docstring} src.data.network.compute_distance_matrix
```
````

````{py:data} __all__
:canonical: src.data.network.__all__
:value: >
   ['DistanceStrategy', 'IterativeDistanceStrategy', 'GoogleMapsStrategy', 'GeoPandasStrategy', 'OSMStr...

```{autodoc2-docstring} src.data.network.__all__
```

````
