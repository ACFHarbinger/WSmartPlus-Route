# {py:mod}`src.configs.must_go`

```{py:module} src.configs.must_go
```

```{autodoc2-docstring} src.configs.must_go
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MustGoConfig <src.configs.must_go.MustGoConfig>`
  - ```{autodoc2-docstring} src.configs.must_go.MustGoConfig
    :summary:
    ```
````

### API

`````{py:class} MustGoConfig
:canonical: src.configs.must_go.MustGoConfig

```{autodoc2-docstring} src.configs.must_go.MustGoConfig
```

````{py:attribute} strategy
:canonical: src.configs.must_go.MustGoConfig.strategy
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.strategy
```

````

````{py:attribute} threshold
:canonical: src.configs.must_go.MustGoConfig.threshold
:type: float
:value: >
   0.7

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.threshold
```

````

````{py:attribute} frequency
:canonical: src.configs.must_go.MustGoConfig.frequency
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.frequency
```

````

````{py:attribute} confidence_factor
:canonical: src.configs.must_go.MustGoConfig.confidence_factor
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.confidence_factor
```

````

````{py:attribute} revenue_kg
:canonical: src.configs.must_go.MustGoConfig.revenue_kg
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.revenue_kg
```

````

````{py:attribute} bin_capacity
:canonical: src.configs.must_go.MustGoConfig.bin_capacity
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.bin_capacity
```

````

````{py:attribute} revenue_threshold
:canonical: src.configs.must_go.MustGoConfig.revenue_threshold
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.revenue_threshold
```

````

````{py:attribute} max_fill
:canonical: src.configs.must_go.MustGoConfig.max_fill
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.max_fill
```

````

````{py:attribute} combined_strategies
:canonical: src.configs.must_go.MustGoConfig.combined_strategies
:type: typing.Optional[list]
:value: >
   None

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.combined_strategies
```

````

````{py:attribute} logic
:canonical: src.configs.must_go.MustGoConfig.logic
:type: str
:value: >
   'or'

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.logic
```

````

````{py:attribute} hidden_dim
:canonical: src.configs.must_go.MustGoConfig.hidden_dim
:type: int
:value: >
   128

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.hidden_dim
```

````

````{py:attribute} lstm_hidden
:canonical: src.configs.must_go.MustGoConfig.lstm_hidden
:type: int
:value: >
   64

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.lstm_hidden
```

````

````{py:attribute} history_length
:canonical: src.configs.must_go.MustGoConfig.history_length
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.history_length
```

````

````{py:attribute} critical_threshold
:canonical: src.configs.must_go.MustGoConfig.critical_threshold
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.critical_threshold
```

````

````{py:attribute} manager_weights
:canonical: src.configs.must_go.MustGoConfig.manager_weights
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.manager_weights
```

````

````{py:attribute} device
:canonical: src.configs.must_go.MustGoConfig.device
:type: str
:value: >
   'cuda'

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.device
```

````

````{py:attribute} params
:canonical: src.configs.must_go.MustGoConfig.params
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.must_go.MustGoConfig.params
```

````

`````
