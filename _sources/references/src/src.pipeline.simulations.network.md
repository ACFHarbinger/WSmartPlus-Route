# {py:mod}`src.pipeline.simulations.network`

```{py:module} src.pipeline.simulations.network
```

```{autodoc2-docstring} src.pipeline.simulations.network
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

src.pipeline.simulations.network.base
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.pipeline.simulations.network.euclidean
src.pipeline.simulations.network.file
src.pipeline.simulations.network.osm
src.pipeline.simulations.network.google
src.pipeline.simulations.network.haversine
src.pipeline.simulations.network.geodesic
src.pipeline.simulations.network.geopandas
```

## Package Contents

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
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.pipeline.simulations.network.__all__>`
  - ```{autodoc2-docstring} src.pipeline.simulations.network.__all__
    :summary:
    ```
````

### API

````{py:function} haversine_distance(lat1: typing.Union[float, numpy.ndarray, pandas.Series], lng1: typing.Union[float, numpy.ndarray, pandas.Series], lat2: typing.Union[float, numpy.ndarray, pandas.Series], lng2: typing.Union[float, numpy.ndarray, pandas.Series]) -> typing.Union[float, numpy.ndarray]
:canonical: src.pipeline.simulations.network.haversine_distance

```{autodoc2-docstring} src.pipeline.simulations.network.haversine_distance
```
````

````{py:function} compute_distance_matrix(coords: pandas.DataFrame, method: str, **kwargs: typing.Any) -> numpy.ndarray
:canonical: src.pipeline.simulations.network.compute_distance_matrix

```{autodoc2-docstring} src.pipeline.simulations.network.compute_distance_matrix
```
````

````{py:data} __all__
:canonical: src.pipeline.simulations.network.__all__
:value: >
   ['DistanceStrategy', 'IterativeDistanceStrategy', 'GoogleMapsStrategy', 'GeoPandasStrategy', 'OSMStr...

```{autodoc2-docstring} src.pipeline.simulations.network.__all__
```

````
