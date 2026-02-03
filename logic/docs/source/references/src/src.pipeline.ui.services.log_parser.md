# {py:mod}`src.pipeline.ui.services.log_parser`

```{py:module} src.pipeline.ui.services.log_parser
```

```{autodoc2-docstring} src.pipeline.ui.services.log_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DayLogEntry <src.pipeline.ui.services.log_parser.DayLogEntry>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.log_parser.DayLogEntry
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`parse_day_log_line <src.pipeline.ui.services.log_parser.parse_day_log_line>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.log_parser.parse_day_log_line
    :summary:
    ```
* - {py:obj}`parse_log_file <src.pipeline.ui.services.log_parser.parse_log_file>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.log_parser.parse_log_file
    :summary:
    ```
* - {py:obj}`stream_log_file <src.pipeline.ui.services.log_parser.stream_log_file>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.log_parser.stream_log_file
    :summary:
    ```
* - {py:obj}`get_unique_policies <src.pipeline.ui.services.log_parser.get_unique_policies>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.log_parser.get_unique_policies
    :summary:
    ```
* - {py:obj}`get_unique_samples <src.pipeline.ui.services.log_parser.get_unique_samples>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.log_parser.get_unique_samples
    :summary:
    ```
* - {py:obj}`filter_entries <src.pipeline.ui.services.log_parser.filter_entries>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.log_parser.filter_entries
    :summary:
    ```
* - {py:obj}`get_day_range <src.pipeline.ui.services.log_parser.get_day_range>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.log_parser.get_day_range
    :summary:
    ```
* - {py:obj}`aggregate_metrics_by_day <src.pipeline.ui.services.log_parser.aggregate_metrics_by_day>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.log_parser.aggregate_metrics_by_day
    :summary:
    ```
````

### API

`````{py:class} DayLogEntry
:canonical: src.pipeline.ui.services.log_parser.DayLogEntry

```{autodoc2-docstring} src.pipeline.ui.services.log_parser.DayLogEntry
```

````{py:attribute} policy
:canonical: src.pipeline.ui.services.log_parser.DayLogEntry.policy
:type: str
:value: >
   None

```{autodoc2-docstring} src.pipeline.ui.services.log_parser.DayLogEntry.policy
```

````

````{py:attribute} sample_id
:canonical: src.pipeline.ui.services.log_parser.DayLogEntry.sample_id
:type: int
:value: >
   None

```{autodoc2-docstring} src.pipeline.ui.services.log_parser.DayLogEntry.sample_id
```

````

````{py:attribute} day
:canonical: src.pipeline.ui.services.log_parser.DayLogEntry.day
:type: int
:value: >
   None

```{autodoc2-docstring} src.pipeline.ui.services.log_parser.DayLogEntry.day
```

````

````{py:attribute} data
:canonical: src.pipeline.ui.services.log_parser.DayLogEntry.data
:type: typing.Dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} src.pipeline.ui.services.log_parser.DayLogEntry.data
```

````

`````

````{py:function} parse_day_log_line(line: str) -> typing.Optional[src.pipeline.ui.services.log_parser.DayLogEntry]
:canonical: src.pipeline.ui.services.log_parser.parse_day_log_line

```{autodoc2-docstring} src.pipeline.ui.services.log_parser.parse_day_log_line
```
````

````{py:function} parse_log_file(file_path: pathlib.Path) -> typing.List[src.pipeline.ui.services.log_parser.DayLogEntry]
:canonical: src.pipeline.ui.services.log_parser.parse_log_file

```{autodoc2-docstring} src.pipeline.ui.services.log_parser.parse_log_file
```
````

````{py:function} stream_log_file(file_path: pathlib.Path, start_line: int = 0) -> typing.Iterator[src.pipeline.ui.services.log_parser.DayLogEntry]
:canonical: src.pipeline.ui.services.log_parser.stream_log_file

```{autodoc2-docstring} src.pipeline.ui.services.log_parser.stream_log_file
```
````

````{py:function} get_unique_policies(entries: typing.List[src.pipeline.ui.services.log_parser.DayLogEntry]) -> typing.List[str]
:canonical: src.pipeline.ui.services.log_parser.get_unique_policies

```{autodoc2-docstring} src.pipeline.ui.services.log_parser.get_unique_policies
```
````

````{py:function} get_unique_samples(entries: typing.List[src.pipeline.ui.services.log_parser.DayLogEntry]) -> typing.List[int]
:canonical: src.pipeline.ui.services.log_parser.get_unique_samples

```{autodoc2-docstring} src.pipeline.ui.services.log_parser.get_unique_samples
```
````

````{py:function} filter_entries(entries: typing.List[src.pipeline.ui.services.log_parser.DayLogEntry], policy: typing.Optional[str] = None, sample_id: typing.Optional[int] = None, day: typing.Optional[int] = None) -> typing.List[src.pipeline.ui.services.log_parser.DayLogEntry]
:canonical: src.pipeline.ui.services.log_parser.filter_entries

```{autodoc2-docstring} src.pipeline.ui.services.log_parser.filter_entries
```
````

````{py:function} get_day_range(entries: typing.List[src.pipeline.ui.services.log_parser.DayLogEntry]) -> typing.Tuple[int, int]
:canonical: src.pipeline.ui.services.log_parser.get_day_range

```{autodoc2-docstring} src.pipeline.ui.services.log_parser.get_day_range
```
````

````{py:function} aggregate_metrics_by_day(entries: typing.List[src.pipeline.ui.services.log_parser.DayLogEntry], policy: typing.Optional[str] = None) -> typing.Dict[int, typing.Dict[str, typing.List[float]]]
:canonical: src.pipeline.ui.services.log_parser.aggregate_metrics_by_day

```{autodoc2-docstring} src.pipeline.ui.services.log_parser.aggregate_metrics_by_day
```
````
