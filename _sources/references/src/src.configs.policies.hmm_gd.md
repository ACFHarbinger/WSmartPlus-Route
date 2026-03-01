# {py:mod}`src.configs.policies.hmm_gd`

```{py:module} src.configs.policies.hmm_gd
```

```{autodoc2-docstring} src.configs.policies.hmm_gd
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HMMGDConfig <src.configs.policies.hmm_gd.HMMGDConfig>`
  - ```{autodoc2-docstring} src.configs.policies.hmm_gd.HMMGDConfig
    :summary:
    ```
````

### API

`````{py:class} HMMGDConfig
:canonical: src.configs.policies.hmm_gd.HMMGDConfig

```{autodoc2-docstring} src.configs.policies.hmm_gd.HMMGDConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.hmm_gd.HMMGDConfig.engine
:type: str
:value: >
   'hmm_gd'

```{autodoc2-docstring} src.configs.policies.hmm_gd.HMMGDConfig.engine
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.hmm_gd.HMMGDConfig.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.hmm_gd.HMMGDConfig.max_iterations
```

````

````{py:attribute} flood_margin
:canonical: src.configs.policies.hmm_gd.HMMGDConfig.flood_margin
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.policies.hmm_gd.HMMGDConfig.flood_margin
```

````

````{py:attribute} rain_speed
:canonical: src.configs.policies.hmm_gd.HMMGDConfig.rain_speed
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.configs.policies.hmm_gd.HMMGDConfig.rain_speed
```

````

````{py:attribute} learning_rate
:canonical: src.configs.policies.hmm_gd.HMMGDConfig.learning_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.hmm_gd.HMMGDConfig.learning_rate
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.hmm_gd.HMMGDConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.hmm_gd.HMMGDConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.hmm_gd.HMMGDConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.hmm_gd.HMMGDConfig.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.hmm_gd.HMMGDConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.hmm_gd.HMMGDConfig.time_limit
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.hmm_gd.HMMGDConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.hmm_gd.HMMGDConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.hmm_gd.HMMGDConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hmm_gd.HMMGDConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.hmm_gd.HMMGDConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hmm_gd.HMMGDConfig.post_processing
```

````

`````
