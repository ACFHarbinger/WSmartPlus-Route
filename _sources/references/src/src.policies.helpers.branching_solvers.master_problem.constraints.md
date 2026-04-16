# {py:mod}`src.policies.helpers.branching_solvers.master_problem.constraints`

```{py:module} src.policies.helpers.branching_solvers.master_problem.constraints
```

```{autodoc2-docstring} src.policies.helpers.branching_solvers.master_problem.constraints
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VRPPMasterProblemConstraintsMixin <src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin>`
  - ```{autodoc2-docstring} src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.helpers.branching_solvers.master_problem.constraints.logger>`
  - ```{autodoc2-docstring} src.policies.helpers.branching_solvers.master_problem.constraints.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.helpers.branching_solvers.master_problem.constraints.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.helpers.branching_solvers.master_problem.constraints.logger
```

````

`````{py:class} VRPPMasterProblemConstraintsMixin
:canonical: src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin

```{autodoc2-docstring} src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin
```

````{py:method} add_edge_clique_cut(u: int, v: int, coefficients: typing.Optional[typing.Dict[int, float]] = None, rhs: float = 1.0) -> bool
:canonical: src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin.add_edge_clique_cut

```{autodoc2-docstring} src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin.add_edge_clique_cut
```

````

````{py:method} add_subset_row_cut(node_set: typing.Union[typing.List[int], typing.Set[int], typing.FrozenSet[int]]) -> bool
:canonical: src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin.add_subset_row_cut

```{autodoc2-docstring} src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin.add_subset_row_cut
```

````

````{py:method} add_capacity_cut(node_list: typing.List[int], rhs: float, coefficients: typing.Optional[typing.Dict[int, float]] = None, is_global: bool = True, _skip_pool: bool = False) -> bool
:canonical: src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin.add_capacity_cut

```{autodoc2-docstring} src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin.add_capacity_cut
```

````

````{py:method} add_lci_cut(node_list: typing.List[int], rhs: float, coefficients: typing.Dict[int, float], node_alphas: typing.Optional[typing.Dict[int, float]] = None, arc: typing.Optional[typing.Tuple[int, int]] = None) -> bool
:canonical: src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin.add_lci_cut

```{autodoc2-docstring} src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin.add_lci_cut
```

````

````{py:method} add_set_packing_capacity_cut(node_list: typing.List[int], rhs: float) -> bool
:canonical: src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin.add_set_packing_capacity_cut

```{autodoc2-docstring} src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin.add_set_packing_capacity_cut
```

````

````{py:method} add_sec_cut(node_list: typing.Union[typing.List[int], typing.Set[int], typing.FrozenSet[int]], rhs: float, cut_name: str = '', global_cut: bool = True, node_i: int = -1, node_j: int = -1, facet_form: str = '2.1') -> bool
:canonical: src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin.add_sec_cut

```{autodoc2-docstring} src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin.add_sec_cut
```

````

````{py:method} _count_crossings(route: src.policies.helpers.branching_solvers.master_problem.model.Route, node_set: typing.FrozenSet[int]) -> int
:canonical: src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin._count_crossings

```{autodoc2-docstring} src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin._count_crossings
```

````

````{py:method} remove_local_cuts() -> int
:canonical: src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin.remove_local_cuts

```{autodoc2-docstring} src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin.remove_local_cuts
```

````

````{py:method} find_and_add_violated_rcc(route_values: typing.Dict[int, float], routes: typing.List[src.policies.helpers.branching_solvers.master_problem.model.Route], max_cuts: int = 5) -> int
:canonical: src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin.find_and_add_violated_rcc

```{autodoc2-docstring} src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin.find_and_add_violated_rcc
```

````

````{py:method} _find_customer_components(arc_flow: typing.Dict[typing.Tuple[int, int], float]) -> typing.List[typing.Set[int]]
:canonical: src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin._find_customer_components

```{autodoc2-docstring} src.policies.helpers.branching_solvers.master_problem.constraints.VRPPMasterProblemConstraintsMixin._find_customer_components
```

````

`````
