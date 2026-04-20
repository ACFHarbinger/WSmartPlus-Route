# {py:mod}`src.configs.policies.mp_sa`

```{py:module} src.configs.policies.mp_sa
```

```{autodoc2-docstring} src.configs.policies.mp_sa
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MP_SA_Config <src.configs.policies.mp_sa.MP_SA_Config>`
  - ```{autodoc2-docstring} src.configs.policies.mp_sa.MP_SA_Config
    :summary:
    ```
````

### API

`````{py:class} MP_SA_Config
:canonical: src.configs.policies.mp_sa.MP_SA_Config

```{autodoc2-docstring} src.configs.policies.mp_sa.MP_SA_Config
```

````{py:attribute} iters
:canonical: src.configs.policies.mp_sa.MP_SA_Config.iters
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.mp_sa.MP_SA_Config.iters
```

````

````{py:attribute} init_temp
:canonical: src.configs.policies.mp_sa.MP_SA_Config.init_temp
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.mp_sa.MP_SA_Config.init_temp
```

````

````{py:attribute} cooling_rate
:canonical: src.configs.policies.mp_sa.MP_SA_Config.cooling_rate
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.mp_sa.MP_SA_Config.cooling_rate
```

````

````{py:attribute} seed
:canonical: src.configs.policies.mp_sa.MP_SA_Config.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.mp_sa.MP_SA_Config.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.mp_sa.MP_SA_Config.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.mp_sa.MP_SA_Config.vrpp
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.mp_sa.MP_SA_Config.mandatory_selection
:type: typing.Optional[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.mp_sa.MP_SA_Config.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.mp_sa.MP_SA_Config.route_improvement
:type: typing.Optional[src.configs.policies.other.route_improvement.RouteImprovingConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.mp_sa.MP_SA_Config.route_improvement
```

````

`````
