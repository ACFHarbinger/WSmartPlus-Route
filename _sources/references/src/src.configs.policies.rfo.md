# {py:mod}`src.configs.policies.rfo`

```{py:module} src.configs.policies.rfo
```

```{autodoc2-docstring} src.configs.policies.rfo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RFOConfig <src.configs.policies.rfo.RFOConfig>`
  - ```{autodoc2-docstring} src.configs.policies.rfo.RFOConfig
    :summary:
    ```
````

### API

`````{py:class} RFOConfig
:canonical: src.configs.policies.rfo.RFOConfig

```{autodoc2-docstring} src.configs.policies.rfo.RFOConfig
```

````{py:attribute} window_size
:canonical: src.configs.policies.rfo.RFOConfig.window_size
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.rfo.RFOConfig.window_size
```

````

````{py:attribute} step_size
:canonical: src.configs.policies.rfo.RFOConfig.step_size
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.rfo.RFOConfig.step_size
```

````

````{py:attribute} mip_time
:canonical: src.configs.policies.rfo.RFOConfig.mip_time
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.rfo.RFOConfig.mip_time
```

````

````{py:attribute} mip_gap
:canonical: src.configs.policies.rfo.RFOConfig.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.rfo.RFOConfig.mip_gap
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.rfo.RFOConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.rfo.RFOConfig.vrpp
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.rfo.RFOConfig.mandatory_selection
:type: typing.Optional[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.rfo.RFOConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.rfo.RFOConfig.route_improvement
:type: typing.Optional[src.configs.policies.other.route_improvement.RouteImprovingConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.rfo.RFOConfig.route_improvement
```

````

`````
