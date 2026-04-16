# {py:mod}`src.configs.policies.bb`

```{py:module} src.configs.policies.bb
```

```{autodoc2-docstring} src.configs.policies.bb
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BBConfig <src.configs.policies.bb.BBConfig>`
  - ```{autodoc2-docstring} src.configs.policies.bb.BBConfig
    :summary:
    ```
````

### API

`````{py:class} BBConfig
:canonical: src.configs.policies.bb.BBConfig

```{autodoc2-docstring} src.configs.policies.bb.BBConfig
```

````{py:attribute} formulation
:canonical: src.configs.policies.bb.BBConfig.formulation
:type: str
:value: >
   'dfj'

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.formulation
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.bb.BBConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.time_limit
```

````

````{py:attribute} mip_gap
:canonical: src.configs.policies.bb.BBConfig.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.mip_gap
```

````

````{py:attribute} branching_strategy
:canonical: src.configs.policies.bb.BBConfig.branching_strategy
:type: str
:value: >
   'strong'

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.branching_strategy
```

````

````{py:attribute} strong_branching_limit
:canonical: src.configs.policies.bb.BBConfig.strong_branching_limit
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.strong_branching_limit
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.bb.BBConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.vrpp
```

````

````{py:attribute} seed
:canonical: src.configs.policies.bb.BBConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.seed
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.bb.BBConfig.mandatory_selection
:type: typing.Optional[typing.List[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.bb.BBConfig.route_improvement
:type: typing.Optional[typing.List[src.configs.policies.other.route_improvement.RouteImprovingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.route_improvement
```

````

````{py:attribute} lr_lambda_init
:canonical: src.configs.policies.bb.BBConfig.lr_lambda_init
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.lr_lambda_init
```

````

````{py:attribute} lr_max_subgradient_iters
:canonical: src.configs.policies.bb.BBConfig.lr_max_subgradient_iters
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.lr_max_subgradient_iters
```

````

````{py:attribute} lr_subgradient_theta
:canonical: src.configs.policies.bb.BBConfig.lr_subgradient_theta
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.lr_subgradient_theta
```

````

````{py:attribute} lr_subgradient_time_fraction
:canonical: src.configs.policies.bb.BBConfig.lr_subgradient_time_fraction
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.lr_subgradient_time_fraction
```

````

````{py:attribute} lr_op_time_limit
:canonical: src.configs.policies.bb.BBConfig.lr_op_time_limit
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.lr_op_time_limit
```

````

````{py:attribute} lr_branching_strategy
:canonical: src.configs.policies.bb.BBConfig.lr_branching_strategy
:type: str
:value: >
   'max_waste'

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.lr_branching_strategy
```

````

````{py:attribute} lr_max_bb_nodes
:canonical: src.configs.policies.bb.BBConfig.lr_max_bb_nodes
:type: int
:value: >
   5000

```{autodoc2-docstring} src.configs.policies.bb.BBConfig.lr_max_bb_nodes
```

````

`````
