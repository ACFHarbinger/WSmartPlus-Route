# {py:mod}`src.utils.logging.modules.analysis`

```{py:module} src.utils.logging.modules.analysis
```

```{autodoc2-docstring} src.utils.logging.modules.analysis
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`load_log_dict <src.utils.logging.modules.analysis.load_log_dict>`
  - ```{autodoc2-docstring} src.utils.logging.modules.analysis.load_log_dict
    :summary:
    ```
* - {py:obj}`output_stats <src.utils.logging.modules.analysis.output_stats>`
  - ```{autodoc2-docstring} src.utils.logging.modules.analysis.output_stats
    :summary:
    ```
* - {py:obj}`runs_per_policy <src.utils.logging.modules.analysis.runs_per_policy>`
  - ```{autodoc2-docstring} src.utils.logging.modules.analysis.runs_per_policy
    :summary:
    ```
* - {py:obj}`final_simulation_summary <src.utils.logging.modules.analysis.final_simulation_summary>`
  - ```{autodoc2-docstring} src.utils.logging.modules.analysis.final_simulation_summary
    :summary:
    ```
````

### API

````{py:function} load_log_dict(dir_paths: typing.List[str], nsamples: typing.List[int], show_incomplete: bool = False, lock: typing.Optional[threading.Lock] = None) -> typing.Dict[str, str]
:canonical: src.utils.logging.modules.analysis.load_log_dict

```{autodoc2-docstring} src.utils.logging.modules.analysis.load_log_dict
```
````

````{py:function} output_stats(dir_path: str, nsamples: int, policies: typing.List[str], keys: typing.List[str], sort_log_func: typing.Optional[typing.Any] = None, print_output: bool = False, lock: typing.Optional[threading.Lock] = None) -> typing.Tuple[typing.Dict[str, typing.Any], typing.Dict[str, typing.Any]]
:canonical: src.utils.logging.modules.analysis.output_stats

```{autodoc2-docstring} src.utils.logging.modules.analysis.output_stats
```
````

````{py:function} runs_per_policy(dir_paths: typing.List[str], nsamples: typing.List[int], policies: typing.List[str], print_output: bool = False, lock: typing.Optional[threading.Lock] = None) -> typing.List[typing.Dict[str, typing.List[int]]]
:canonical: src.utils.logging.modules.analysis.runs_per_policy

```{autodoc2-docstring} src.utils.logging.modules.analysis.runs_per_policy
```
````

````{py:function} final_simulation_summary(log: typing.Dict[str, typing.Any], policy: str, n_samples: int) -> None
:canonical: src.utils.logging.modules.analysis.final_simulation_summary

```{autodoc2-docstring} src.utils.logging.modules.analysis.final_simulation_summary
```
````
