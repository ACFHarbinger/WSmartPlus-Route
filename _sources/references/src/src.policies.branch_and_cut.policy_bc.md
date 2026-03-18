# {py:mod}`src.policies.branch_and_cut.policy_bc`

```{py:module} src.policies.branch_and_cut.policy_bc
```

```{autodoc2-docstring} src.policies.branch_and_cut.policy_bc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolicyBC <src.policies.branch_and_cut.policy_bc.PolicyBC>`
  - ```{autodoc2-docstring} src.policies.branch_and_cut.policy_bc.PolicyBC
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_branch_and_cut <src.policies.branch_and_cut.policy_bc.run_branch_and_cut>`
  - ```{autodoc2-docstring} src.policies.branch_and_cut.policy_bc.run_branch_and_cut
    :summary:
    ```
````

### API

`````{py:class} PolicyBC(time_limit: float = 60.0, mip_gap: float = 0.01, max_cuts_per_round: int = 50, use_heuristics: bool = True, verbose: bool = False)
:canonical: src.policies.branch_and_cut.policy_bc.PolicyBC

```{autodoc2-docstring} src.policies.branch_and_cut.policy_bc.PolicyBC
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_cut.policy_bc.PolicyBC.__init__
```

````{py:method} __call__(coords: pandas.DataFrame, must_go: typing.List[int], distance_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, **kwargs) -> typing.Tuple[typing.List[int], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.branch_and_cut.policy_bc.PolicyBC.__call__

```{autodoc2-docstring} src.policies.branch_and_cut.policy_bc.PolicyBC.__call__
```

````

`````

````{py:function} run_branch_and_cut(coords: pandas.DataFrame, must_go: typing.List[int], distance_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, time_limit: float = 60.0, mip_gap: float = 0.01, verbose: bool = False, **kwargs) -> typing.Tuple[typing.List[int], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.branch_and_cut.policy_bc.run_branch_and_cut

```{autodoc2-docstring} src.policies.branch_and_cut.policy_bc.run_branch_and_cut
```
````
