# {py:mod}`src.pipeline.simulations.solver_status`

```{py:module} src.pipeline.simulations.solver_status
```

```{autodoc2-docstring} src.pipeline.simulations.solver_status
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`reset_solver_status <src.pipeline.simulations.solver_status.reset_solver_status>`
  - ```{autodoc2-docstring} src.pipeline.simulations.solver_status.reset_solver_status
    :summary:
    ```
* - {py:obj}`note_solver_status <src.pipeline.simulations.solver_status.note_solver_status>`
  - ```{autodoc2-docstring} src.pipeline.simulations.solver_status.note_solver_status
    :summary:
    ```
* - {py:obj}`current_solver_status <src.pipeline.simulations.solver_status.current_solver_status>`
  - ```{autodoc2-docstring} src.pipeline.simulations.solver_status.current_solver_status
    :summary:
    ```
* - {py:obj}`format_backend_status <src.pipeline.simulations.solver_status.format_backend_status>`
  - ```{autodoc2-docstring} src.pipeline.simulations.solver_status.format_backend_status
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GUROBI_STATUS_NAMES <src.pipeline.simulations.solver_status.GUROBI_STATUS_NAMES>`
  - ```{autodoc2-docstring} src.pipeline.simulations.solver_status.GUROBI_STATUS_NAMES
    :summary:
    ```
* - {py:obj}`ORTOOLS_STATUS_NAMES <src.pipeline.simulations.solver_status.ORTOOLS_STATUS_NAMES>`
  - ```{autodoc2-docstring} src.pipeline.simulations.solver_status.ORTOOLS_STATUS_NAMES
    :summary:
    ```
* - {py:obj}`_solver_status <src.pipeline.simulations.solver_status._solver_status>`
  - ```{autodoc2-docstring} src.pipeline.simulations.solver_status._solver_status
    :summary:
    ```
````

### API

````{py:data} GUROBI_STATUS_NAMES
:canonical: src.pipeline.simulations.solver_status.GUROBI_STATUS_NAMES
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.solver_status.GUROBI_STATUS_NAMES
```

````

````{py:data} ORTOOLS_STATUS_NAMES
:canonical: src.pipeline.simulations.solver_status.ORTOOLS_STATUS_NAMES
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.solver_status.ORTOOLS_STATUS_NAMES
```

````

````{py:data} _solver_status
:canonical: src.pipeline.simulations.solver_status._solver_status
:type: contextvars.ContextVar[typing.Optional[str]]
:value: >
   'ContextVar(...)'

```{autodoc2-docstring} src.pipeline.simulations.solver_status._solver_status
```

````

````{py:function} reset_solver_status() -> None
:canonical: src.pipeline.simulations.solver_status.reset_solver_status

```{autodoc2-docstring} src.pipeline.simulations.solver_status.reset_solver_status
```
````

````{py:function} note_solver_status(status: typing.Optional[str], append: bool = False) -> None
:canonical: src.pipeline.simulations.solver_status.note_solver_status

```{autodoc2-docstring} src.pipeline.simulations.solver_status.note_solver_status
```
````

````{py:function} current_solver_status() -> typing.Optional[str]
:canonical: src.pipeline.simulations.solver_status.current_solver_status

```{autodoc2-docstring} src.pipeline.simulations.solver_status.current_solver_status
```
````

````{py:function} format_backend_status(backend: str, code: object) -> str
:canonical: src.pipeline.simulations.solver_status.format_backend_status

```{autodoc2-docstring} src.pipeline.simulations.solver_status.format_backend_status
```
````
