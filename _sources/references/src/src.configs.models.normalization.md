# {py:mod}`src.configs.models.normalization`

```{py:module} src.configs.models.normalization
```

```{autodoc2-docstring} src.configs.models.normalization
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NormalizationConfig <src.configs.models.normalization.NormalizationConfig>`
  - ```{autodoc2-docstring} src.configs.models.normalization.NormalizationConfig
    :summary:
    ```
````

### API

`````{py:class} NormalizationConfig
:canonical: src.configs.models.normalization.NormalizationConfig

```{autodoc2-docstring} src.configs.models.normalization.NormalizationConfig
```

````{py:attribute} norm_type
:canonical: src.configs.models.normalization.NormalizationConfig.norm_type
:type: str
:value: >
   'batch'

```{autodoc2-docstring} src.configs.models.normalization.NormalizationConfig.norm_type
```

````

````{py:attribute} epsilon
:canonical: src.configs.models.normalization.NormalizationConfig.epsilon
:type: float
:value: >
   None

```{autodoc2-docstring} src.configs.models.normalization.NormalizationConfig.epsilon
```

````

````{py:attribute} learn_affine
:canonical: src.configs.models.normalization.NormalizationConfig.learn_affine
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.models.normalization.NormalizationConfig.learn_affine
```

````

````{py:attribute} track_stats
:canonical: src.configs.models.normalization.NormalizationConfig.track_stats
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.models.normalization.NormalizationConfig.track_stats
```

````

````{py:attribute} momentum
:canonical: src.configs.models.normalization.NormalizationConfig.momentum
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.models.normalization.NormalizationConfig.momentum
```

````

````{py:attribute} n_groups
:canonical: src.configs.models.normalization.NormalizationConfig.n_groups
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.models.normalization.NormalizationConfig.n_groups
```

````

````{py:attribute} k_lrnorm
:canonical: src.configs.models.normalization.NormalizationConfig.k_lrnorm
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.models.normalization.NormalizationConfig.k_lrnorm
```

````

`````
