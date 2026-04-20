# {py:mod}`src.configs.policies.mp_aco`

```{py:module} src.configs.policies.mp_aco
```

```{autodoc2-docstring} src.configs.policies.mp_aco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MP_ACO_Config <src.configs.policies.mp_aco.MP_ACO_Config>`
  - ```{autodoc2-docstring} src.configs.policies.mp_aco.MP_ACO_Config
    :summary:
    ```
````

### API

`````{py:class} MP_ACO_Config
:canonical: src.configs.policies.mp_aco.MP_ACO_Config

```{autodoc2-docstring} src.configs.policies.mp_aco.MP_ACO_Config
```

````{py:attribute} n_ants
:canonical: src.configs.policies.mp_aco.MP_ACO_Config.n_ants
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.mp_aco.MP_ACO_Config.n_ants
```

````

````{py:attribute} iters
:canonical: src.configs.policies.mp_aco.MP_ACO_Config.iters
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.mp_aco.MP_ACO_Config.iters
```

````

````{py:attribute} alpha
:canonical: src.configs.policies.mp_aco.MP_ACO_Config.alpha
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.mp_aco.MP_ACO_Config.alpha
```

````

````{py:attribute} beta
:canonical: src.configs.policies.mp_aco.MP_ACO_Config.beta
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.mp_aco.MP_ACO_Config.beta
```

````

````{py:attribute} rho
:canonical: src.configs.policies.mp_aco.MP_ACO_Config.rho
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.mp_aco.MP_ACO_Config.rho
```

````

````{py:attribute} seed
:canonical: src.configs.policies.mp_aco.MP_ACO_Config.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.mp_aco.MP_ACO_Config.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.mp_aco.MP_ACO_Config.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.mp_aco.MP_ACO_Config.vrpp
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.mp_aco.MP_ACO_Config.mandatory_selection
:type: typing.Optional[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.mp_aco.MP_ACO_Config.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.mp_aco.MP_ACO_Config.route_improvement
:type: typing.Optional[src.configs.policies.other.route_improvement.RouteImprovingConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.mp_aco.MP_ACO_Config.route_improvement
```

````

`````
