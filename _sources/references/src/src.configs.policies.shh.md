# {py:mod}`src.configs.policies.shh`

```{py:module} src.configs.policies.shh
```

```{autodoc2-docstring} src.configs.policies.shh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SHHConfig <src.configs.policies.shh.SHHConfig>`
  - ```{autodoc2-docstring} src.configs.policies.shh.SHHConfig
    :summary:
    ```
````

### API

`````{py:class} SHHConfig
:canonical: src.configs.policies.shh.SHHConfig

```{autodoc2-docstring} src.configs.policies.shh.SHHConfig
```

````{py:attribute} iters
:canonical: src.configs.policies.shh.SHHConfig.iters
:type: int
:value: >
   200

```{autodoc2-docstring} src.configs.policies.shh.SHHConfig.iters
```

````

````{py:attribute} history_len
:canonical: src.configs.policies.shh.SHHConfig.history_len
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.shh.SHHConfig.history_len
```

````

````{py:attribute} seed
:canonical: src.configs.policies.shh.SHHConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.shh.SHHConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.shh.SHHConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.shh.SHHConfig.vrpp
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.shh.SHHConfig.mandatory_selection
:type: typing.Optional[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.shh.SHHConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.shh.SHHConfig.route_improvement
:type: typing.Optional[src.configs.policies.other.route_improvement.RouteImprovingConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.shh.SHHConfig.route_improvement
```

````

`````
