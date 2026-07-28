# {py:mod}`src.tracking.logging.modules.analysis`

```{py:module} src.tracking.logging.modules.analysis
```

```{autodoc2-docstring} src.tracking.logging.modules.analysis
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ResultsDB <src.tracking.logging.modules.analysis.ResultsDB>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.analysis.ResultsDB
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_parse_slug <src.tracking.logging.modules.analysis._parse_slug>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.analysis._parse_slug
    :summary:
    ```
* - {py:obj}`load_log_dict <src.tracking.logging.modules.analysis.load_log_dict>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.analysis.load_log_dict
    :summary:
    ```
* - {py:obj}`output_stats <src.tracking.logging.modules.analysis.output_stats>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.analysis.output_stats
    :summary:
    ```
* - {py:obj}`runs_per_policy <src.tracking.logging.modules.analysis.runs_per_policy>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.analysis.runs_per_policy
    :summary:
    ```
* - {py:obj}`final_simulation_summary <src.tracking.logging.modules.analysis.final_simulation_summary>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.analysis.final_simulation_summary
    :summary:
    ```
* - {py:obj}`display_simulation_summary_table <src.tracking.logging.modules.analysis.display_simulation_summary_table>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.analysis.display_simulation_summary_table
    :summary:
    ```
* - {py:obj}`display_per_policy_simulation_summary <src.tracking.logging.modules.analysis.display_per_policy_simulation_summary>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.analysis.display_per_policy_simulation_summary
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_MANDATORY_PREFIXES <src.tracking.logging.modules.analysis._MANDATORY_PREFIXES>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.analysis._MANDATORY_PREFIXES
    :summary:
    ```
* - {py:obj}`_ROUTE_IMPROVERS <src.tracking.logging.modules.analysis._ROUTE_IMPROVERS>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.analysis._ROUTE_IMPROVERS
    :summary:
    ```
* - {py:obj}`_ROUTE_CONSTRUCTORS <src.tracking.logging.modules.analysis._ROUTE_CONSTRUCTORS>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.analysis._ROUTE_CONSTRUCTORS
    :summary:
    ```
* - {py:obj}`_ROUTE_CONSTRUCTOR_ENGINES <src.tracking.logging.modules.analysis._ROUTE_CONSTRUCTOR_ENGINES>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.analysis._ROUTE_CONSTRUCTOR_ENGINES
    :summary:
    ```
* - {py:obj}`_ACCEPTANCE_CRITERIA <src.tracking.logging.modules.analysis._ACCEPTANCE_CRITERIA>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.analysis._ACCEPTANCE_CRITERIA
    :summary:
    ```
````

### API

````{py:data} _MANDATORY_PREFIXES
:canonical: src.tracking.logging.modules.analysis._MANDATORY_PREFIXES
:type: typing.List[str]
:value: >
   ['last_minute_cf70', 'last_minute_cf80', 'last_minute_cf90', 'last_minute_cf', 'service_level1', 'se...

```{autodoc2-docstring} src.tracking.logging.modules.analysis._MANDATORY_PREFIXES
```

````

````{py:data} _ROUTE_IMPROVERS
:canonical: src.tracking.logging.modules.analysis._ROUTE_IMPROVERS
:type: typing.List[str]
:value: >
   ['tsp_fast_tsp', 'cvrp_ortools', 'ftsp', 'tsp']

```{autodoc2-docstring} src.tracking.logging.modules.analysis._ROUTE_IMPROVERS
```

````

````{py:data} _ROUTE_CONSTRUCTORS
:canonical: src.tracking.logging.modules.analysis._ROUTE_CONSTRUCTORS
:type: typing.List[str]
:value: >
   ['vrpp_gurobi', 'pg_clns', 'swc_tcf', 'sans_new', 'ks_aco', 'aco_hh', 'alns', 'psoma', 'sisr', 'hvpl...

```{autodoc2-docstring} src.tracking.logging.modules.analysis._ROUTE_CONSTRUCTORS
```

````

````{py:data} _ROUTE_CONSTRUCTOR_ENGINES
:canonical: src.tracking.logging.modules.analysis._ROUTE_CONSTRUCTOR_ENGINES
:type: typing.List[str]
:value: >
   []

```{autodoc2-docstring} src.tracking.logging.modules.analysis._ROUTE_CONSTRUCTOR_ENGINES
```

````

````{py:data} _ACCEPTANCE_CRITERIA
:canonical: src.tracking.logging.modules.analysis._ACCEPTANCE_CRITERIA
:type: typing.List[str]
:value: >
   []

```{autodoc2-docstring} src.tracking.logging.modules.analysis._ACCEPTANCE_CRITERIA
```

````

````{py:function} _parse_slug(slug: str, mandatory_prefixes: typing.Optional[typing.List[str]] = None, route_constructors: typing.Optional[typing.List[str]] = None, route_improvers: typing.Optional[typing.List[str]] = None, route_constructor_engines: typing.Optional[typing.List[str]] = None, acceptance_criteria: typing.Optional[typing.List[str]] = None) -> typing.Dict[str, str]
:canonical: src.tracking.logging.modules.analysis._parse_slug

```{autodoc2-docstring} src.tracking.logging.modules.analysis._parse_slug
```
````

`````{py:class} ResultsDB(df: typing.Any)
:canonical: src.tracking.logging.modules.analysis.ResultsDB

```{autodoc2-docstring} src.tracking.logging.modules.analysis.ResultsDB
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.logging.modules.analysis.ResultsDB.__init__
```

````{py:attribute} METADATA_COLS
:canonical: src.tracking.logging.modules.analysis.ResultsDB.METADATA_COLS
:type: typing.List[str]
:value: >
   ['n_days', 'n_bins', 'area', 'waste_type', 'data_distribution', 'run_name', 'policy_slug', 'n_sample...

```{autodoc2-docstring} src.tracking.logging.modules.analysis.ResultsDB.METADATA_COLS
```

````

````{py:method} from_records(records: typing.List[typing.Dict[str, typing.Any]]) -> src.tracking.logging.modules.analysis.ResultsDB
:canonical: src.tracking.logging.modules.analysis.ResultsDB.from_records
:classmethod:

```{autodoc2-docstring} src.tracking.logging.modules.analysis.ResultsDB.from_records
```

````

````{py:method} filter(n_days: typing.Optional[typing.Any] = None, n_bins: typing.Optional[typing.Any] = None, area: typing.Optional[str] = None, waste_type: typing.Optional[str] = None, data_distribution: typing.Optional[str] = None, run_name: typing.Optional[str] = None, policy_slug: typing.Optional[str] = None, mandatory_selection: typing.Optional[str] = None, route_constructor: typing.Optional[str] = None, route_constructor_engine: typing.Optional[str] = None, acceptance_criterion: typing.Optional[str] = None, route_improver: typing.Optional[str] = None) -> src.tracking.logging.modules.analysis.ResultsDB
:canonical: src.tracking.logging.modules.analysis.ResultsDB.filter

```{autodoc2-docstring} src.tracking.logging.modules.analysis.ResultsDB.filter
```

````

````{py:method} to_dataframe(include_raw: bool = False) -> typing.Any
:canonical: src.tracking.logging.modules.analysis.ResultsDB.to_dataframe

```{autodoc2-docstring} src.tracking.logging.modules.analysis.ResultsDB.to_dataframe
```

````

````{py:property} metric_cols
:canonical: src.tracking.logging.modules.analysis.ResultsDB.metric_cols
:type: typing.List[str]

```{autodoc2-docstring} src.tracking.logging.modules.analysis.ResultsDB.metric_cols
```

````

````{py:property} std_cols
:canonical: src.tracking.logging.modules.analysis.ResultsDB.std_cols
:type: typing.List[str]

```{autodoc2-docstring} src.tracking.logging.modules.analysis.ResultsDB.std_cols
```

````

````{py:method} mean_metrics() -> typing.Any
:canonical: src.tracking.logging.modules.analysis.ResultsDB.mean_metrics

```{autodoc2-docstring} src.tracking.logging.modules.analysis.ResultsDB.mean_metrics
```

````

````{py:method} std_metrics() -> typing.Any
:canonical: src.tracking.logging.modules.analysis.ResultsDB.std_metrics

```{autodoc2-docstring} src.tracking.logging.modules.analysis.ResultsDB.std_metrics
```

````

````{py:method} groupby(by: typing.Any) -> typing.Any
:canonical: src.tracking.logging.modules.analysis.ResultsDB.groupby

```{autodoc2-docstring} src.tracking.logging.modules.analysis.ResultsDB.groupby
```

````

````{py:method} __len__() -> int
:canonical: src.tracking.logging.modules.analysis.ResultsDB.__len__

```{autodoc2-docstring} src.tracking.logging.modules.analysis.ResultsDB.__len__
```

````

````{py:method} __repr__() -> str
:canonical: src.tracking.logging.modules.analysis.ResultsDB.__repr__

````

`````

````{py:function} load_log_dict(home_dir: str, output_dir: str = 'output', *, mandatory_prefixes: typing.Optional[typing.List[str]] = None, route_constructors: typing.Optional[typing.List[str]] = None, route_improvers: typing.Optional[typing.List[str]] = None, route_constructor_engines: typing.Optional[typing.List[str]] = None, acceptance_criteria: typing.Optional[typing.List[str]] = None, load_samples: bool = False, load_daily: bool = False, lock: typing.Optional[threading.Lock] = None) -> src.tracking.logging.modules.analysis.ResultsDB
:canonical: src.tracking.logging.modules.analysis.load_log_dict

```{autodoc2-docstring} src.tracking.logging.modules.analysis.load_log_dict
```
````

````{py:function} output_stats(dir_path: str, nsamples: int, policies: typing.List[str], keys: typing.List[str], sort_log_func: typing.Optional[typing.Any] = None, print_output: bool = False, lock: typing.Optional[threading.Lock] = None) -> typing.Tuple[typing.Dict[str, typing.Any], typing.Dict[str, typing.Any]]
:canonical: src.tracking.logging.modules.analysis.output_stats

```{autodoc2-docstring} src.tracking.logging.modules.analysis.output_stats
```
````

````{py:function} runs_per_policy(dir_paths: typing.List[str], nsamples: typing.List[int], policies: typing.List[str], print_output: bool = False, lock: typing.Optional[threading.Lock] = None) -> typing.List[typing.Dict[str, typing.List[int]]]
:canonical: src.tracking.logging.modules.analysis.runs_per_policy

```{autodoc2-docstring} src.tracking.logging.modules.analysis.runs_per_policy
```
````

````{py:function} final_simulation_summary(log: typing.Dict[str, typing.Any], policy: str, n_samples: int) -> None
:canonical: src.tracking.logging.modules.analysis.final_simulation_summary

```{autodoc2-docstring} src.tracking.logging.modules.analysis.final_simulation_summary
```
````

````{py:function} display_simulation_summary_table(log: typing.Dict[str, typing.Any], title: str = 'Simulation Summary', lock: typing.Optional[typing.Any] = None) -> None
:canonical: src.tracking.logging.modules.analysis.display_simulation_summary_table

```{autodoc2-docstring} src.tracking.logging.modules.analysis.display_simulation_summary_table
```
````

````{py:function} display_per_policy_simulation_summary(pol_name: str, sample_id: int, aggregate_metrics: typing.List[float], daily_log: typing.Dict[str, typing.List[typing.Any]], title_prefix: str = 'Results for', lock: typing.Optional[typing.Any] = None) -> None
:canonical: src.tracking.logging.modules.analysis.display_per_policy_simulation_summary

```{autodoc2-docstring} src.tracking.logging.modules.analysis.display_per_policy_simulation_summary
```
````
