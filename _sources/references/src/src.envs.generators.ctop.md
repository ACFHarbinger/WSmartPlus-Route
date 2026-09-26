# {py:mod}`src.envs.generators.ctop`

```{py:module} src.envs.generators.ctop
```

```{autodoc2-docstring} src.envs.generators.ctop
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CTOPGenerator <src.envs.generators.ctop.CTOPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.ctop.CTOPGenerator
    :summary:
    ```
````

### API

`````{py:class} CTOPGenerator(*args, shift_hours: typing.Union[float, None] = None, avg_speed_kmh: typing.Union[float, None] = None, service_time_h: typing.Union[float, None] = None, **kwargs)
:canonical: src.envs.generators.ctop.CTOPGenerator

Bases: {py:obj}`logic.src.envs.generators.vrpp.VRPPGenerator`

```{autodoc2-docstring} src.envs.generators.ctop.CTOPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.ctop.CTOPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.ctop.CTOPGenerator._generate

```{autodoc2-docstring} src.envs.generators.ctop.CTOPGenerator._generate
```

````

`````
