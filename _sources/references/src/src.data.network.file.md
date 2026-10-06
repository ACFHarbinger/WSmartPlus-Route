# {py:mod}`src.data.network.file`

```{py:module} src.data.network.file
```

```{autodoc2-docstring} src.data.network.file
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FileStrategy <src.data.network.file.FileStrategy>`
  - ```{autodoc2-docstring} src.data.network.file.FileStrategy
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_bin_id <src.data.network.file._bin_id>`
  - ```{autodoc2-docstring} src.data.network.file._bin_id
    :summary:
    ```
* - {py:obj}`_matrix_for_bin_ids <src.data.network.file._matrix_for_bin_ids>`
  - ```{autodoc2-docstring} src.data.network.file._matrix_for_bin_ids
    :summary:
    ```
````

### API

`````{py:class} FileStrategy
:canonical: src.data.network.file.FileStrategy

Bases: {py:obj}`src.data.network.base.DistanceStrategy`

```{autodoc2-docstring} src.data.network.file.FileStrategy
```

````{py:method} calculate(coords: pandas.DataFrame, **kwargs: typing.Any) -> numpy.ndarray
:canonical: src.data.network.file.FileStrategy.calculate

```{autodoc2-docstring} src.data.network.file.FileStrategy.calculate
```

````

`````

````{py:function} _bin_id(value: typing.Any) -> typing.Any
:canonical: src.data.network.file._bin_id

```{autodoc2-docstring} src.data.network.file._bin_id
```
````

````{py:function} _matrix_for_bin_ids(distance_matrix: numpy.ndarray, matrix_ids: numpy.ndarray, req_ids: numpy.ndarray) -> numpy.ndarray
:canonical: src.data.network.file._matrix_for_bin_ids

```{autodoc2-docstring} src.data.network.file._matrix_for_bin_ids
```
````
