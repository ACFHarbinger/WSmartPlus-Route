# {py:mod}`src.utils.target.remover`

```{py:module} src.utils.target.remover
```

```{autodoc2-docstring} src.utils.target.remover
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`remove_from_json_file <src.utils.target.remover.remove_from_json_file>`
  - ```{autodoc2-docstring} src.utils.target.remover.remove_from_json_file
    :summary:
    ```
* - {py:obj}`remove_from_jsonl_file <src.utils.target.remover.remove_from_jsonl_file>`
  - ```{autodoc2-docstring} src.utils.target.remover.remove_from_jsonl_file
    :summary:
    ```
* - {py:obj}`remove_checkpoint_files <src.utils.target.remover.remove_checkpoint_files>`
  - ```{autodoc2-docstring} src.utils.target.remover.remove_checkpoint_files
    :summary:
    ```
* - {py:obj}`remove_fill_history_files <src.utils.target.remover.remove_fill_history_files>`
  - ```{autodoc2-docstring} src.utils.target.remover.remove_fill_history_files
    :summary:
    ```
* - {py:obj}`remove_targeted_runs <src.utils.target.remover.remove_targeted_runs>`
  - ```{autodoc2-docstring} src.utils.target.remover.remove_targeted_runs
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.utils.target.remover.__all__>`
  - ```{autodoc2-docstring} src.utils.target.remover.__all__
    :summary:
    ```
* - {py:obj}`_JSONL_PREFIX_RE <src.utils.target.remover._JSONL_PREFIX_RE>`
  - ```{autodoc2-docstring} src.utils.target.remover._JSONL_PREFIX_RE
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: src.utils.target.remover.__all__
:value: >
   ['PolicyFilter', 'remove_targeted_runs', 'remove_from_json_file', 'remove_from_jsonl_file', 'remove_...

```{autodoc2-docstring} src.utils.target.remover.__all__
```

````

````{py:function} remove_from_json_file(path: str, policy_filter: logic.src.utils.target.matcher.PolicyFilter, dry_run: bool = False) -> typing.List[str]
:canonical: src.utils.target.remover.remove_from_json_file

```{autodoc2-docstring} src.utils.target.remover.remove_from_json_file
```
````

````{py:data} _JSONL_PREFIX_RE
:canonical: src.utils.target.remover._JSONL_PREFIX_RE
:value: >
   'compile(...)'

```{autodoc2-docstring} src.utils.target.remover._JSONL_PREFIX_RE
```

````

````{py:function} remove_from_jsonl_file(path: str, policy_filter: logic.src.utils.target.matcher.PolicyFilter, dry_run: bool = False) -> typing.List[str]
:canonical: src.utils.target.remover.remove_from_jsonl_file

```{autodoc2-docstring} src.utils.target.remover.remove_from_jsonl_file
```
````

````{py:function} remove_checkpoint_files(checkpoints_dir: str, policy_filter: logic.src.utils.target.matcher.PolicyFilter, dry_run: bool = False) -> typing.List[str]
:canonical: src.utils.target.remover.remove_checkpoint_files

```{autodoc2-docstring} src.utils.target.remover.remove_checkpoint_files
```
````

````{py:function} remove_fill_history_files(fill_history_dir: str, policy_filter: logic.src.utils.target.matcher.PolicyFilter, dry_run: bool = False) -> typing.List[str]
:canonical: src.utils.target.remover.remove_fill_history_files

```{autodoc2-docstring} src.utils.target.remover.remove_fill_history_files
```
````

````{py:function} remove_targeted_runs(results_dir: str, policy_filter: logic.src.utils.target.matcher.PolicyFilter, distributions: typing.Optional[typing.List[str]] = None, dry_run: bool = False, verbose: bool = True) -> typing.List[str]
:canonical: src.utils.target.remover.remove_targeted_runs

```{autodoc2-docstring} src.utils.target.remover.remove_targeted_runs
```
````
