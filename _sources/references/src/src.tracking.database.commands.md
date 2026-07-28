# {py:mod}`src.tracking.database.commands`

```{py:module} src.tracking.database.commands
```

```{autodoc2-docstring} src.tracking.database.commands
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`inspect_database <src.tracking.database.commands.inspect_database>`
  - ```{autodoc2-docstring} src.tracking.database.commands.inspect_database
    :summary:
    ```
* - {py:obj}`clean_database <src.tracking.database.commands.clean_database>`
  - ```{autodoc2-docstring} src.tracking.database.commands.clean_database
    :summary:
    ```
* - {py:obj}`compact_database <src.tracking.database.commands.compact_database>`
  - ```{autodoc2-docstring} src.tracking.database.commands.compact_database
    :summary:
    ```
* - {py:obj}`prune_database <src.tracking.database.commands.prune_database>`
  - ```{autodoc2-docstring} src.tracking.database.commands.prune_database
    :summary:
    ```
* - {py:obj}`_resolve_run_id <src.tracking.database.commands._resolve_run_id>`
  - ```{autodoc2-docstring} src.tracking.database.commands._resolve_run_id
    :summary:
    ```
* - {py:obj}`export_run <src.tracking.database.commands.export_run>`
  - ```{autodoc2-docstring} src.tracking.database.commands.export_run
    :summary:
    ```
````

### API

````{py:function} inspect_database() -> None
:canonical: src.tracking.database.commands.inspect_database

```{autodoc2-docstring} src.tracking.database.commands.inspect_database
```
````

````{py:function} clean_database() -> None
:canonical: src.tracking.database.commands.clean_database

```{autodoc2-docstring} src.tracking.database.commands.clean_database
```
````

````{py:function} compact_database() -> None
:canonical: src.tracking.database.commands.compact_database

```{autodoc2-docstring} src.tracking.database.commands.compact_database
```
````

````{py:function} prune_database(older_than_days: int = 30, status: str = 'failed', experiment_name: str = '', dry_run: bool = False) -> None
:canonical: src.tracking.database.commands.prune_database

```{autodoc2-docstring} src.tracking.database.commands.prune_database
```
````

````{py:function} _resolve_run_id(conn: sqlite3.Connection, run_id: str, experiment_name: str, latest: bool) -> str
:canonical: src.tracking.database.commands._resolve_run_id

```{autodoc2-docstring} src.tracking.database.commands._resolve_run_id
```
````

````{py:function} export_run(run_id: str = '', experiment_name: str = '', latest: bool = False, output: str = '') -> None
:canonical: src.tracking.database.commands.export_run

```{autodoc2-docstring} src.tracking.database.commands.export_run
```
````
