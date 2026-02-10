# {py:mod}`src.configs.policies.vrpp`

```{py:module} src.configs.policies.vrpp
```

```{autodoc2-docstring} src.configs.policies.vrpp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VRPPConfig <src.configs.policies.vrpp.VRPPConfig>`
  - ```{autodoc2-docstring} src.configs.policies.vrpp.VRPPConfig
    :summary:
    ```
````

### API

`````{py:class} VRPPConfig
:canonical: src.configs.policies.vrpp.VRPPConfig

```{autodoc2-docstring} src.configs.policies.vrpp.VRPPConfig
```

````{py:attribute} Omega
:canonical: src.configs.policies.vrpp.VRPPConfig.Omega
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.vrpp.VRPPConfig.Omega
```

````

````{py:attribute} delta
:canonical: src.configs.policies.vrpp.VRPPConfig.delta
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.vrpp.VRPPConfig.delta
```

````

````{py:attribute} psi
:canonical: src.configs.policies.vrpp.VRPPConfig.psi
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.vrpp.VRPPConfig.psi
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.vrpp.VRPPConfig.time_limit
:type: float
:value: >
   600.0

```{autodoc2-docstring} src.configs.policies.vrpp.VRPPConfig.time_limit
```

````

````{py:attribute} engine
:canonical: src.configs.policies.vrpp.VRPPConfig.engine
:type: str
:value: >
   'gurobi'

```{autodoc2-docstring} src.configs.policies.vrpp.VRPPConfig.engine
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.vrpp.VRPPConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.vrpp.VRPPConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.vrpp.VRPPConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.vrpp.VRPPConfig.post_processing
```

````

`````
