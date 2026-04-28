# {py:mod}`src.tracking.logging.modules.analysis`

```{py:module} src.tracking.logging.modules.analysis
```

```{autodoc2-docstring} src.tracking.logging.modules.analysis
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

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

### API

````{py:function} load_log_dict(dir_paths: typing.List[str], nsamples: typing.List[int], show_incomplete: bool = False, lock: typing.Optional[threading.Lock] = None) -> typing.Dict[str, str]
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
