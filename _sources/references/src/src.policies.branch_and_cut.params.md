# {py:mod}`src.policies.branch_and_cut.params`

```{py:module} src.policies.branch_and_cut.params
```

```{autodoc2-docstring} src.policies.branch_and_cut.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BCParams <src.policies.branch_and_cut.params.BCParams>`
  - ```{autodoc2-docstring} src.policies.branch_and_cut.params.BCParams
    :summary:
    ```
````

### API

`````{py:class} BCParams
:canonical: src.policies.branch_and_cut.params.BCParams

```{autodoc2-docstring} src.policies.branch_and_cut.params.BCParams
```

````{py:attribute} time_limit
:canonical: src.policies.branch_and_cut.params.BCParams.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.policies.branch_and_cut.params.BCParams.time_limit
```

````

````{py:attribute} mip_gap
:canonical: src.policies.branch_and_cut.params.BCParams.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.branch_and_cut.params.BCParams.mip_gap
```

````

````{py:attribute} max_cuts_per_round
:canonical: src.policies.branch_and_cut.params.BCParams.max_cuts_per_round
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.branch_and_cut.params.BCParams.max_cuts_per_round
```

````

````{py:attribute} use_heuristics
:canonical: src.policies.branch_and_cut.params.BCParams.use_heuristics
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.branch_and_cut.params.BCParams.use_heuristics
```

````

````{py:attribute} verbose
:canonical: src.policies.branch_and_cut.params.BCParams.verbose
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_cut.params.BCParams.verbose
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.branch_and_cut.params.BCParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_cut.params.BCParams.profit_aware_operators
```

````

````{py:attribute} vrpp
:canonical: src.policies.branch_and_cut.params.BCParams.vrpp
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_cut.params.BCParams.vrpp
```

````

````{py:attribute} enable_fractional_capacity_cuts
:canonical: src.policies.branch_and_cut.params.BCParams.enable_fractional_capacity_cuts
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.branch_and_cut.params.BCParams.enable_fractional_capacity_cuts
```

````

````{py:attribute} use_comb_cuts
:canonical: src.policies.branch_and_cut.params.BCParams.use_comb_cuts
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_cut.params.BCParams.use_comb_cuts
```

````

````{py:method} from_config(config: typing.Dict[str, typing.Any]) -> src.policies.branch_and_cut.params.BCParams
:canonical: src.policies.branch_and_cut.params.BCParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.branch_and_cut.params.BCParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.branch_and_cut.params.BCParams.to_dict

```{autodoc2-docstring} src.policies.branch_and_cut.params.BCParams.to_dict
```

````

`````
