# {py:mod}`src.policies.other.branching_solvers.master_problem.pool`

```{py:module} src.policies.other.branching_solvers.master_problem.pool
```

```{autodoc2-docstring} src.policies.other.branching_solvers.master_problem.pool
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CutInfo <src.policies.other.branching_solvers.master_problem.pool.CutInfo>`
  - ```{autodoc2-docstring} src.policies.other.branching_solvers.master_problem.pool.CutInfo
    :summary:
    ```
* - {py:obj}`GlobalCutPool <src.policies.other.branching_solvers.master_problem.pool.GlobalCutPool>`
  - ```{autodoc2-docstring} src.policies.other.branching_solvers.master_problem.pool.GlobalCutPool
    :summary:
    ```
````

### API

`````{py:class} CutInfo
:canonical: src.policies.other.branching_solvers.master_problem.pool.CutInfo

```{autodoc2-docstring} src.policies.other.branching_solvers.master_problem.pool.CutInfo
```

````{py:attribute} type
:canonical: src.policies.other.branching_solvers.master_problem.pool.CutInfo.type
:type: str
:value: >
   None

```{autodoc2-docstring} src.policies.other.branching_solvers.master_problem.pool.CutInfo.type
```

````

````{py:attribute} data
:canonical: src.policies.other.branching_solvers.master_problem.pool.CutInfo.data
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} src.policies.other.branching_solvers.master_problem.pool.CutInfo.data
```

````

````{py:attribute} active
:canonical: src.policies.other.branching_solvers.master_problem.pool.CutInfo.active
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.other.branching_solvers.master_problem.pool.CutInfo.active
```

````

````{py:attribute} violation
:canonical: src.policies.other.branching_solvers.master_problem.pool.CutInfo.violation
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.other.branching_solvers.master_problem.pool.CutInfo.violation
```

````

`````

`````{py:class} GlobalCutPool()
:canonical: src.policies.other.branching_solvers.master_problem.pool.GlobalCutPool

```{autodoc2-docstring} src.policies.other.branching_solvers.master_problem.pool.GlobalCutPool
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.branching_solvers.master_problem.pool.GlobalCutPool.__init__
```

````{py:method} add_cut(cut_type: str, data: typing.Any) -> None
:canonical: src.policies.other.branching_solvers.master_problem.pool.GlobalCutPool.add_cut

```{autodoc2-docstring} src.policies.other.branching_solvers.master_problem.pool.GlobalCutPool.add_cut
```

````

````{py:method} apply_to_master(master: src.policies.other.branching_solvers.master_problem.problem_support.MasterProblemSupport) -> int
:canonical: src.policies.other.branching_solvers.master_problem.pool.GlobalCutPool.apply_to_master

```{autodoc2-docstring} src.policies.other.branching_solvers.master_problem.pool.GlobalCutPool.apply_to_master
```

````

`````
