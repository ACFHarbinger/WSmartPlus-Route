# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_branching`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_branching
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_branching
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ScenarioConsistentBranching <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_branching.ScenarioConsistentBranching>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_branching.ScenarioConsistentBranching
    :summary:
    ```
````

### API

`````{py:class} ScenarioConsistentBranching(base_threshold: float = 0.95)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_branching.ScenarioConsistentBranching

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_branching.ScenarioConsistentBranching
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_branching.ScenarioConsistentBranching.__init__
```

````{py:method} calculate_consensus(var_id: int, scenario_tree: typing.Any, deterministic_scenario_solutions: typing.Dict[int, typing.Dict[int, int]]) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_branching.ScenarioConsistentBranching.calculate_consensus

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_branching.ScenarioConsistentBranching.calculate_consensus
```

````

````{py:method} select_branching_variable(fractional_vars: typing.List[typing.Any], scenario_tree: typing.Any, deterministic_solutions: typing.Dict[int, typing.Dict[int, int]], days_remaining: int, initial_horizon: int) -> typing.Any
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_branching.ScenarioConsistentBranching.select_branching_variable

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_branching.ScenarioConsistentBranching.select_branching_variable
```

````

`````
