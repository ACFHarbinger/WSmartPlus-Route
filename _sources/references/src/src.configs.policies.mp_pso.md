# {py:mod}`src.configs.policies.mp_pso`

```{py:module} src.configs.policies.mp_pso
```

```{autodoc2-docstring} src.configs.policies.mp_pso
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MP_PSO_Config <src.configs.policies.mp_pso.MP_PSO_Config>`
  - ```{autodoc2-docstring} src.configs.policies.mp_pso.MP_PSO_Config
    :summary:
    ```
````

### API

`````{py:class} MP_PSO_Config
:canonical: src.configs.policies.mp_pso.MP_PSO_Config

```{autodoc2-docstring} src.configs.policies.mp_pso.MP_PSO_Config
```

````{py:attribute} swarm_size
:canonical: src.configs.policies.mp_pso.MP_PSO_Config.swarm_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.mp_pso.MP_PSO_Config.swarm_size
```

````

````{py:attribute} iters
:canonical: src.configs.policies.mp_pso.MP_PSO_Config.iters
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.mp_pso.MP_PSO_Config.iters
```

````

````{py:attribute} w
:canonical: src.configs.policies.mp_pso.MP_PSO_Config.w
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.configs.policies.mp_pso.MP_PSO_Config.w
```

````

````{py:attribute} c1
:canonical: src.configs.policies.mp_pso.MP_PSO_Config.c1
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.mp_pso.MP_PSO_Config.c1
```

````

````{py:attribute} c2
:canonical: src.configs.policies.mp_pso.MP_PSO_Config.c2
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.mp_pso.MP_PSO_Config.c2
```

````

````{py:attribute} seed
:canonical: src.configs.policies.mp_pso.MP_PSO_Config.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.mp_pso.MP_PSO_Config.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.mp_pso.MP_PSO_Config.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.mp_pso.MP_PSO_Config.vrpp
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.mp_pso.MP_PSO_Config.mandatory_selection
:type: typing.Optional[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.mp_pso.MP_PSO_Config.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.mp_pso.MP_PSO_Config.route_improvement
:type: typing.Optional[src.configs.policies.other.route_improvement.RouteImprovingConfig]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.mp_pso.MP_PSO_Config.route_improvement
```

````

`````
