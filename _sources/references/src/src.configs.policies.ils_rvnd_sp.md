# {py:mod}`src.configs.policies.ils_rvnd_sp`

```{py:module} src.configs.policies.ils_rvnd_sp
```

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ILSRVNDSPConfig <src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig
    :summary:
    ```
````

### API

`````{py:class} ILSRVNDSPConfig
:canonical: src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig

Bases: {py:obj}`src.configs.policies.abc.ABCConfig`

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig
```

````{py:attribute} max_restarts
:canonical: src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.max_restarts
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.max_restarts
```

````

````{py:attribute} max_iter_ils
:canonical: src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.max_iter_ils
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.max_iter_ils
```

````

````{py:attribute} perturbation_strength
:canonical: src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.perturbation_strength
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.perturbation_strength
```

````

````{py:attribute} use_set_partitioning
:canonical: src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.use_set_partitioning
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.use_set_partitioning
```

````

````{py:attribute} mip_time_limit
:canonical: src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.mip_time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.mip_time_limit
```

````

````{py:attribute} sp_mip_gap
:canonical: src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.sp_mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.sp_mip_gap
```

````

````{py:attribute} N
:canonical: src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.N
:type: int
:value: >
   150

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.N
```

````

````{py:attribute} A
:canonical: src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.A
:type: float
:value: >
   11.0

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.A
```

````

````{py:attribute} MaxIter_a
:canonical: src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.MaxIter_a
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.MaxIter_a
```

````

````{py:attribute} MaxIter_b
:canonical: src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.MaxIter_b
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.MaxIter_b
```

````

````{py:attribute} MaxIterILS_b
:canonical: src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.MaxIterILS_b
:type: int
:value: >
   2000

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.MaxIterILS_b
```

````

````{py:attribute} TDev_a
:canonical: src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.TDev_a
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.TDev_a
```

````

````{py:attribute} TDev_b
:canonical: src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.TDev_b
:type: float
:value: >
   0.005

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.TDev_b
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.time_limit
:type: float
:value: >
   120.0

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.ils_rvnd_sp.ILSRVNDSPConfig.seed
```

````

`````
