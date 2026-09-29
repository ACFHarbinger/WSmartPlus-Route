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

src.data.network.geopandas
src.data.network.osm
src.data.network.file
src.data.network.google
src.data.network.euclidean
src.data.network.geodesic
src.data.network.haversine
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
* - {py:obj}`_strategy_class <src.data.network._strategy_class>`
  - ```{autodoc2-docstring} src.data.network._strategy_class
    :summary:
    ```
* - {py:obj}`__getattr__ <src.data.network.__getattr__>`
  - ```{autodoc2-docstring} src.data.network.__getattr__
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_OPTIONAL_STRATEGIES <src.data.network._OPTIONAL_STRATEGIES>`
  - ```{autodoc2-docstring} src.data.network._OPTIONAL_STRATEGIES
    :summary:
    ```
* - {py:obj}`_METHOD_TO_ATTR <src.data.network._METHOD_TO_ATTR>`
  - ```{autodoc2-docstring} src.data.network._METHOD_TO_ATTR
    :summary:
    ```
* - {py:obj}`__all__ <src.data.network.__all__>`
  - ```{autodoc2-docstring} src.data.network.__all__
    :summary:
    ```
````

### API

````{py:data} _OPTIONAL_STRATEGIES
:canonical: src.data.network._OPTIONAL_STRATEGIES
:value: >
   None

```{autodoc2-docstring} src.data.network._OPTIONAL_STRATEGIES
```

````

````{py:data} _METHOD_TO_ATTR
:canonical: src.data.network._METHOD_TO_ATTR
:value: >
   None

```{autodoc2-docstring} src.data.network._METHOD_TO_ATTR
```

````

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

````{py:function} _strategy_class(method: str) -> typing.Any
:canonical: src.data.network._strategy_class

```{autodoc2-docstring} src.data.network._strategy_class
```
````

````{py:function} __getattr__(name: str) -> typing.Any
:canonical: src.data.network.__getattr__

```{autodoc2-docstring} src.data.network.__getattr__
```
````

````{py:data} __all__
:canonical: src.data.network.__all__
:value: >
   ['DistanceStrategy', 'IterativeDistanceStrategy', 'GoogleMapsStrategy', 'GeoPandasStrategy', 'OSMStr...

```{autodoc2-docstring} src.data.network.__all__
```

````
