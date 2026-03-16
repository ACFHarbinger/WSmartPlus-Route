# {py:mod}`src.configs.policies.hmm_gd_hh`

```{py:module} src.configs.policies.hmm_gd_hh
```

```{autodoc2-docstring} src.configs.policies.hmm_gd_hh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HMMGDHHConfig <src.configs.policies.hmm_gd_hh.HMMGDHHConfig>`
  - ```{autodoc2-docstring} src.configs.policies.hmm_gd_hh.HMMGDHHConfig
    :summary:
    ```
````

### API

`````{py:class} HMMGDHHConfig
:canonical: src.configs.policies.hmm_gd_hh.HMMGDHHConfig

```{autodoc2-docstring} src.configs.policies.hmm_gd_hh.HMMGDHHConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.hmm_gd_hh.HMMGDHHConfig.engine
:type: str
:value: >
   'hmm_gd_hh'

```{autodoc2-docstring} src.configs.policies.hmm_gd_hh.HMMGDHHConfig.engine
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.hmm_gd_hh.HMMGDHHConfig.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.hmm_gd_hh.HMMGDHHConfig.max_iterations
```

````

````{py:attribute} flood_margin
:canonical: src.configs.policies.hmm_gd_hh.HMMGDHHConfig.flood_margin
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.policies.hmm_gd_hh.HMMGDHHConfig.flood_margin
```

````

````{py:attribute} rain_speed
:canonical: src.configs.policies.hmm_gd_hh.HMMGDHHConfig.rain_speed
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.configs.policies.hmm_gd_hh.HMMGDHHConfig.rain_speed
```

````

````{py:attribute} learning_rate
:canonical: src.configs.policies.hmm_gd_hh.HMMGDHHConfig.learning_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.hmm_gd_hh.HMMGDHHConfig.learning_rate
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.hmm_gd_hh.HMMGDHHConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.hmm_gd_hh.HMMGDHHConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.hmm_gd_hh.HMMGDHHConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.hmm_gd_hh.HMMGDHHConfig.n_llh
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.hmm_gd_hh.HMMGDHHConfig.local_search_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.hmm_gd_hh.HMMGDHHConfig.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.hmm_gd_hh.HMMGDHHConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.hmm_gd_hh.HMMGDHHConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.hmm_gd_hh.HMMGDHHConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hmm_gd_hh.HMMGDHHConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.hmm_gd_hh.HMMGDHHConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.hmm_gd_hh.HMMGDHHConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.hmm_gd_hh.HMMGDHHConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hmm_gd_hh.HMMGDHHConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.hmm_gd_hh.HMMGDHHConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hmm_gd_hh.HMMGDHHConfig.post_processing
```

````

`````
