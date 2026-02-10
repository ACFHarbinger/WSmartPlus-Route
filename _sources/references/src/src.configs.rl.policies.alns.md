# {py:mod}`src.configs.rl.policies.alns`

```{py:module} src.configs.rl.policies.alns
```

```{autodoc2-docstring} src.configs.rl.policies.alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ALNSConfig <src.configs.rl.policies.alns.ALNSConfig>`
  - ```{autodoc2-docstring} src.configs.rl.policies.alns.ALNSConfig
    :summary:
    ```
````

### API

`````{py:class} ALNSConfig
:canonical: src.configs.rl.policies.alns.ALNSConfig

```{autodoc2-docstring} src.configs.rl.policies.alns.ALNSConfig
```

````{py:attribute} time_limit
:canonical: src.configs.rl.policies.alns.ALNSConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.rl.policies.alns.ALNSConfig.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.configs.rl.policies.alns.ALNSConfig.max_iterations
:type: int
:value: >
   5000

```{autodoc2-docstring} src.configs.rl.policies.alns.ALNSConfig.max_iterations
```

````

````{py:attribute} start_temp
:canonical: src.configs.rl.policies.alns.ALNSConfig.start_temp
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.rl.policies.alns.ALNSConfig.start_temp
```

````

````{py:attribute} cooling_rate
:canonical: src.configs.rl.policies.alns.ALNSConfig.cooling_rate
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.configs.rl.policies.alns.ALNSConfig.cooling_rate
```

````

````{py:attribute} reaction_factor
:canonical: src.configs.rl.policies.alns.ALNSConfig.reaction_factor
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.rl.policies.alns.ALNSConfig.reaction_factor
```

````

````{py:attribute} min_removal
:canonical: src.configs.rl.policies.alns.ALNSConfig.min_removal
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.rl.policies.alns.ALNSConfig.min_removal
```

````

````{py:attribute} max_removal_pct
:canonical: src.configs.rl.policies.alns.ALNSConfig.max_removal_pct
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.configs.rl.policies.alns.ALNSConfig.max_removal_pct
```

````

`````
