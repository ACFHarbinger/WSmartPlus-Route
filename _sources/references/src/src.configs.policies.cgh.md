# {py:mod}`src.configs.policies.cgh`

```{py:module} src.configs.policies.cgh
```

```{autodoc2-docstring} src.configs.policies.cgh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CGHConfig <src.configs.policies.cgh.CGHConfig>`
  - ```{autodoc2-docstring} src.configs.policies.cgh.CGHConfig
    :summary:
    ```
````

### API

`````{py:class} CGHConfig
:canonical: src.configs.policies.cgh.CGHConfig

```{autodoc2-docstring} src.configs.policies.cgh.CGHConfig
```

````{py:attribute} cg_iters
:canonical: src.configs.policies.cgh.CGHConfig.cg_iters
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.cgh.CGHConfig.cg_iters
```

````

````{py:attribute} routes_per_iter
:canonical: src.configs.policies.cgh.CGHConfig.routes_per_iter
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.cgh.CGHConfig.routes_per_iter
```

````

````{py:attribute} seed
:canonical: src.configs.policies.cgh.CGHConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.cgh.CGHConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.cgh.CGHConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.cgh.CGHConfig.vrpp
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.cgh.CGHConfig.mandatory_selection
:type: typing.Optional[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.cgh.CGHConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.cgh.CGHConfig.route_improvement
:type: typing.Optional[src.configs.policies.other.route_improvement.RouteImprovingConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.cgh.CGHConfig.route_improvement
```

````

`````
