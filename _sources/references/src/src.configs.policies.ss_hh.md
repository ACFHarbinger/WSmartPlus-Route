# {py:mod}`src.configs.policies.ss_hh`

```{py:module} src.configs.policies.ss_hh
```

```{autodoc2-docstring} src.configs.policies.ss_hh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SSHHConfig <src.configs.policies.ss_hh.SSHHConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ss_hh.SSHHConfig
    :summary:
    ```
````

### API

`````{py:class} SSHHConfig
:canonical: src.configs.policies.ss_hh.SSHHConfig

```{autodoc2-docstring} src.configs.policies.ss_hh.SSHHConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.ss_hh.SSHHConfig.engine
:type: str
:value: >
   'ss_hh'

```{autodoc2-docstring} src.configs.policies.ss_hh.SSHHConfig.engine
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.ss_hh.SSHHConfig.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.ss_hh.SSHHConfig.max_iterations
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.ss_hh.SSHHConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.ss_hh.SSHHConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.ss_hh.SSHHConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.ss_hh.SSHHConfig.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.ss_hh.SSHHConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.ss_hh.SSHHConfig.time_limit
```

````

````{py:attribute} threshold_infeasible
:canonical: src.configs.policies.ss_hh.SSHHConfig.threshold_infeasible
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.configs.policies.ss_hh.SSHHConfig.threshold_infeasible
```

````

````{py:attribute} threshold_feasible_base
:canonical: src.configs.policies.ss_hh.SSHHConfig.threshold_feasible_base
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.configs.policies.ss_hh.SSHHConfig.threshold_feasible_base
```

````

````{py:attribute} threshold_decay_rate
:canonical: src.configs.policies.ss_hh.SSHHConfig.threshold_decay_rate
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.ss_hh.SSHHConfig.threshold_decay_rate
```

````

````{py:attribute} seed
:canonical: src.configs.policies.ss_hh.SSHHConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ss_hh.SSHHConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.ss_hh.SSHHConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ss_hh.SSHHConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.ss_hh.SSHHConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ss_hh.SSHHConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.ss_hh.SSHHConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ss_hh.SSHHConfig.post_processing
```

````

`````
