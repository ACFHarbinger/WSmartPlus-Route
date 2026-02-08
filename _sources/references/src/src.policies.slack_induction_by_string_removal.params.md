# {py:mod}`src.policies.slack_induction_by_string_removal.params`

```{py:module} src.policies.slack_induction_by_string_removal.params
```

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SISRParams <src.policies.slack_induction_by_string_removal.params.SISRParams>`
  - ```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.params.SISRParams
    :summary:
    ```
````

### API

`````{py:class} SISRParams
:canonical: src.policies.slack_induction_by_string_removal.params.SISRParams

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.params.SISRParams
```

````{py:attribute} time_limit
:canonical: src.policies.slack_induction_by_string_removal.params.SISRParams.time_limit
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.params.SISRParams.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.policies.slack_induction_by_string_removal.params.SISRParams.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.params.SISRParams.max_iterations
```

````

````{py:attribute} start_temp
:canonical: src.policies.slack_induction_by_string_removal.params.SISRParams.start_temp
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.params.SISRParams.start_temp
```

````

````{py:attribute} cooling_rate
:canonical: src.policies.slack_induction_by_string_removal.params.SISRParams.cooling_rate
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.params.SISRParams.cooling_rate
```

````

````{py:attribute} max_string_len
:canonical: src.policies.slack_induction_by_string_removal.params.SISRParams.max_string_len
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.params.SISRParams.max_string_len
```

````

````{py:attribute} avg_string_len
:canonical: src.policies.slack_induction_by_string_removal.params.SISRParams.avg_string_len
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.params.SISRParams.avg_string_len
```

````

````{py:attribute} blink_rate
:canonical: src.policies.slack_induction_by_string_removal.params.SISRParams.blink_rate
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.params.SISRParams.blink_rate
```

````

````{py:attribute} destroy_ratio
:canonical: src.policies.slack_induction_by_string_removal.params.SISRParams.destroy_ratio
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.params.SISRParams.destroy_ratio
```

````

````{py:method} from_config(config: logic.src.configs.policies.SISRConfig) -> src.policies.slack_induction_by_string_removal.params.SISRParams
:canonical: src.policies.slack_induction_by_string_removal.params.SISRParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.params.SISRParams.from_config
```

````

`````
