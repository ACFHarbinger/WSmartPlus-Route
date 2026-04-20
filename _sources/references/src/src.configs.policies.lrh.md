# {py:mod}`src.configs.policies.lrh`

```{py:module} src.configs.policies.lrh
```

```{autodoc2-docstring} src.configs.policies.lrh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LRHConfig <src.configs.policies.lrh.LRHConfig>`
  - ```{autodoc2-docstring} src.configs.policies.lrh.LRHConfig
    :summary:
    ```
````

### API

`````{py:class} LRHConfig
:canonical: src.configs.policies.lrh.LRHConfig

```{autodoc2-docstring} src.configs.policies.lrh.LRHConfig
```

````{py:attribute} max_iter
:canonical: src.configs.policies.lrh.LRHConfig.max_iter
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.lrh.LRHConfig.max_iter
```

````

````{py:attribute} step_size
:canonical: src.configs.policies.lrh.LRHConfig.step_size
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.lrh.LRHConfig.step_size
```

````

````{py:attribute} halving_freq
:canonical: src.configs.policies.lrh.LRHConfig.halving_freq
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.lrh.LRHConfig.halving_freq
```

````

````{py:attribute} seed
:canonical: src.configs.policies.lrh.LRHConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.lrh.LRHConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.lrh.LRHConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.lrh.LRHConfig.vrpp
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.lrh.LRHConfig.mandatory_selection
:type: typing.Optional[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.lrh.LRHConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.lrh.LRHConfig.route_improvement
:type: typing.Optional[src.configs.policies.other.route_improvement.RouteImprovingConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.lrh.LRHConfig.route_improvement
```

````

`````
