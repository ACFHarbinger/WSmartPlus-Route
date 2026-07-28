# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`STEFParams <src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams
    :summary:
    ```
````

### API

`````{py:class} STEFParams
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams
```

````{py:attribute} num_days
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.num_days
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.num_days
```

````

````{py:attribute} num_realizations
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.num_realizations
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.num_realizations
```

````

````{py:attribute} mean_increment
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.mean_increment
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.mean_increment
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.time_limit
```

````

````{py:attribute} mip_gap
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.mip_gap
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.mip_gap
```

````

````{py:attribute} waste_weight
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.waste_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.waste_weight
```

````

````{py:attribute} cost_weight
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.cost_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.cost_weight
```

````

````{py:attribute} overflow_penalty
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.overflow_penalty
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.overflow_penalty
```

````

````{py:attribute} discount_factor
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.discount_factor
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.discount_factor
```

````

````{py:attribute} use_mtz
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.use_mtz
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.use_mtz
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.seed
:type: typing.Optional[int]
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.seed
```

````

````{py:method} from_config(config: typing.Dict[str, typing.Any]) -> src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.params.STEFParams.to_dict
```

````

`````
