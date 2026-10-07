# {py:mod}`src.envs.generators.tcmvptp`

```{py:module} src.envs.generators.tcmvptp
```

```{autodoc2-docstring} src.envs.generators.tcmvptp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TCMVPTPGenerator <src.envs.generators.tcmvptp.TCMVPTPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.tcmvptp.TCMVPTPGenerator
    :summary:
    ```
````

### API

`````{py:class} TCMVPTPGenerator(*args, shift_hours: typing.Union[float, None] = None, avg_speed_kmh: typing.Union[float, None] = None, service_time_h: typing.Union[float, None] = None, **kwargs)
:canonical: src.envs.generators.tcmvptp.TCMVPTPGenerator

Bases: {py:obj}`logic.src.envs.generators.ptp.PTPGenerator`

```{autodoc2-docstring} src.envs.generators.tcmvptp.TCMVPTPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.tcmvptp.TCMVPTPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.tcmvptp.TCMVPTPGenerator._generate

```{autodoc2-docstring} src.envs.generators.tcmvptp.TCMVPTPGenerator._generate
```

````

`````
