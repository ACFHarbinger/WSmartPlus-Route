# {py:mod}`src.pipeline.simulations.repository.filesystem`

```{py:module} src.pipeline.simulations.repository.filesystem
```

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FileSystemRepository <src.pipeline.simulations.repository.filesystem.FileSystemRepository>`
  - ```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository
    :summary:
    ```
````

### API

`````{py:class} FileSystemRepository(data_root_dir)
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository

Bases: {py:obj}`src.pipeline.simulations.repository.base.SimulationRepository`

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository.__init__
```

````{py:method} _get_data_dir(override_dir: typing.Optional[str] = None) -> str
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository._get_data_dir

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository._get_data_dir
```

````

````{py:method} get_indices(filename: typing.Any, n_samples: int, n_nodes: int, data_size: int, lock: typing.Optional[typing.Any] = None) -> typing.List[typing.List[int]]
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository.get_indices

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository.get_indices
```

````

````{py:method} get_depot(area: typing.Any, data_dir: typing.Optional[str] = None) -> pandas.DataFrame
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository.get_depot

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository.get_depot
```

````

````{py:method} get_simulator_data(number_of_bins: int, area: str = 'Rio Maior', waste_type: typing.Optional[str] = None, lock: typing.Optional[typing.Any] = None, data_dir: typing.Optional[str] = None) -> typing.Tuple[pandas.DataFrame, pandas.DataFrame]
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository.get_simulator_data

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository.get_simulator_data
```

````

````{py:method} _get_mixrmbac_data(d_dir: str, number_of_bins: int, src_area: str) -> typing.Tuple[pandas.DataFrame, pandas.DataFrame]
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository._get_mixrmbac_data

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository._get_mixrmbac_data
```

````

````{py:method} _get_riomaior_data(d_dir: str, number_of_bins: int, src_area: str, wtype: typing.Optional[str]) -> typing.Tuple[pandas.DataFrame, pandas.DataFrame]
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository._get_riomaior_data

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository._get_riomaior_data
```

````

````{py:method} _get_figueiradafoz_data(d_dir: str, number_of_bins: int, src_area: str, wtype: typing.Optional[str]) -> typing.Tuple[pandas.DataFrame, pandas.DataFrame]
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository._get_figueiradafoz_data

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository._get_figueiradafoz_data
```

````

````{py:method} _get_both_areas_data(d_dir: str, number_of_bins: int, src_area: str) -> typing.Tuple[pandas.DataFrame, pandas.DataFrame]
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository._get_both_areas_data

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository._get_both_areas_data
```

````

````{py:method} _preprocess_county_date(data: pandas.DataFrame, date_str: str = 'Date') -> pandas.DataFrame
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository._preprocess_county_date

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository._preprocess_county_date
```

````

````{py:method} _preprocess_county_data(data: pandas.DataFrame) -> pandas.DataFrame
:canonical: src.pipeline.simulations.repository.filesystem.FileSystemRepository._preprocess_county_data

```{autodoc2-docstring} src.pipeline.simulations.repository.filesystem.FileSystemRepository._preprocess_county_data
```

````

`````
