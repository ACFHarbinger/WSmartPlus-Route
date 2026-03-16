# {py:mod}`src.configs.policies.swc_tcf`

```{py:module} src.configs.policies.swc_tcf
```

```{autodoc2-docstring} src.configs.policies.swc_tcf
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SWCTCFConfig <src.configs.policies.swc_tcf.SWCTCFConfig>`
  - ```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig
    :summary:
    ```
````

### API

`````{py:class} SWCTCFConfig
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig
```

````{py:attribute} Omega
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.Omega
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.Omega
```

````

````{py:attribute} delta
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.delta
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.delta
```

````

````{py:attribute} psi
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.psi
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.psi
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.time_limit
:type: float
:value: >
   600.0

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.seed
```

````

````{py:attribute} engine
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.engine
:type: str
:value: >
   'gurobi'

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.engine
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.post_processing
```

````

`````
