# {py:mod}`src.policies.mandatory_selection.selection_kmeans_sector`

```{py:module} src.policies.mandatory_selection.selection_kmeans_sector
```

```{autodoc2-docstring} src.policies.mandatory_selection.selection_kmeans_sector
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KMeansGeographicSectorSelection <src.policies.mandatory_selection.selection_kmeans_sector.KMeansGeographicSectorSelection>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_kmeans_sector.KMeansGeographicSectorSelection
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_fit_kmeans <src.policies.mandatory_selection.selection_kmeans_sector._fit_kmeans>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_kmeans_sector._fit_kmeans
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_CLUSTER_CACHE <src.policies.mandatory_selection.selection_kmeans_sector._CLUSTER_CACHE>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.selection_kmeans_sector._CLUSTER_CACHE
    :summary:
    ```
````

### API

````{py:data} _CLUSTER_CACHE
:canonical: src.policies.mandatory_selection.selection_kmeans_sector._CLUSTER_CACHE
:type: typing.Dict[typing.Tuple[int, int, str], numpy.ndarray]
:value: >
   None

```{autodoc2-docstring} src.policies.mandatory_selection.selection_kmeans_sector._CLUSTER_CACHE
```

````

````{py:function} _fit_kmeans(coordinates: numpy.ndarray, n_sectors: int) -> numpy.ndarray
:canonical: src.policies.mandatory_selection.selection_kmeans_sector._fit_kmeans

```{autodoc2-docstring} src.policies.mandatory_selection.selection_kmeans_sector._fit_kmeans
```
````

`````{py:class} KMeansGeographicSectorSelection
:canonical: src.policies.mandatory_selection.selection_kmeans_sector.KMeansGeographicSectorSelection

Bases: {py:obj}`logic.src.interfaces.mandatory_selection.IMandatorySelectionStrategy`

```{autodoc2-docstring} src.policies.mandatory_selection.selection_kmeans_sector.KMeansGeographicSectorSelection
```

````{py:method} select_bins(context: logic.src.interfaces.context.SelectionContext) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.SearchContext]
:canonical: src.policies.mandatory_selection.selection_kmeans_sector.KMeansGeographicSectorSelection.select_bins

```{autodoc2-docstring} src.policies.mandatory_selection.selection_kmeans_sector.KMeansGeographicSectorSelection.select_bins
```

````

`````
