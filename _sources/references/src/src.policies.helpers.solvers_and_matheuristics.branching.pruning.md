# {py:mod}`src.policies.helpers.solvers_and_matheuristics.branching.pruning`

```{py:module} src.policies.helpers.solvers_and_matheuristics.branching.pruning
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.pruning
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`reset_master_constraints <src.policies.helpers.solvers_and_matheuristics.branching.pruning.reset_master_constraints>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.pruning.reset_master_constraints
    :summary:
    ```
* - {py:obj}`apply_route_level_branching_filters <src.policies.helpers.solvers_and_matheuristics.branching.pruning.apply_route_level_branching_filters>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.pruning.apply_route_level_branching_filters
    :summary:
    ```
* - {py:obj}`apply_branching_to_master <src.policies.helpers.solvers_and_matheuristics.branching.pruning.apply_branching_to_master>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.pruning.apply_branching_to_master
    :summary:
    ```
* - {py:obj}`perform_strong_branching <src.policies.helpers.solvers_and_matheuristics.branching.pruning.perform_strong_branching>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.pruning.perform_strong_branching
    :summary:
    ```
* - {py:obj}`extract_forced_sets_from_constraints <src.policies.helpers.solvers_and_matheuristics.branching.pruning.extract_forced_sets_from_constraints>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.pruning.extract_forced_sets_from_constraints
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.helpers.solvers_and_matheuristics.branching.pruning.logger>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.pruning.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.pruning.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.pruning.logger
```

````

````{py:exception} BPCPruningException()
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.pruning.BPCPruningException

Bases: {py:obj}`Exception`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.pruning.BPCPruningException
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.pruning.BPCPruningException.__init__
```

````

````{py:function} reset_master_constraints(master: logic.src.policies.helpers.solvers_and_matheuristics.VRPPMasterProblem) -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.pruning.reset_master_constraints

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.pruning.reset_master_constraints
```
````

````{py:function} apply_route_level_branching_filters(master: logic.src.policies.helpers.solvers_and_matheuristics.VRPPMasterProblem, bc: logic.src.policies.helpers.solvers_and_matheuristics.AnyBranchingConstraint) -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.pruning.apply_route_level_branching_filters

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.pruning.apply_route_level_branching_filters
```
````

````{py:function} apply_branching_to_master(master: logic.src.policies.helpers.solvers_and_matheuristics.VRPPMasterProblem, branching_constraints: typing.List[logic.src.policies.helpers.solvers_and_matheuristics.AnyBranchingConstraint], branching_strategy: str = 'divergence') -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.pruning.apply_branching_to_master

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.pruning.apply_branching_to_master
```
````

````{py:function} perform_strong_branching(master: logic.src.policies.helpers.solvers_and_matheuristics.VRPPMasterProblem, candidates: typing.List[typing.Tuple[int, typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]], float]], current_node: typing.Optional[logic.src.policies.helpers.solvers_and_matheuristics.BranchNode] = None, strong_branching_size: int = 5) -> typing.Optional[typing.Tuple[int, typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]], float]]
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.pruning.perform_strong_branching

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.pruning.perform_strong_branching
```
````

````{py:function} extract_forced_sets_from_constraints(branching_constraints: typing.Optional[typing.List[logic.src.policies.helpers.solvers_and_matheuristics.AnyBranchingConstraint]]) -> typing.Tuple[typing.Set[int], typing.Set[int]]
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.pruning.extract_forced_sets_from_constraints

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.pruning.extract_forced_sets_from_constraints
```
````
