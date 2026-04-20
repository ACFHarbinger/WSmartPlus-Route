# {py:mod}`src.envs.routing.spctsp`

```{py:module} src.envs.routing.spctsp
```

```{autodoc2-docstring} src.envs.routing.spctsp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SPCTSPEnv <src.envs.routing.spctsp.SPCTSPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.spctsp.SPCTSPEnv
    :summary:
    ```
````

### API

`````{py:class} SPCTSPEnv(generator: typing.Optional[logic.src.envs.generators.pctsp.PCTSPGenerator] = None, generator_params: typing.Optional[dict] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.spctsp.SPCTSPEnv

Bases: {py:obj}`logic.src.envs.routing.pctsp.PCTSPEnv`

```{autodoc2-docstring} src.envs.routing.spctsp.SPCTSPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.spctsp.SPCTSPEnv.__init__
```

````{py:attribute} NAME
:canonical: src.envs.routing.spctsp.SPCTSPEnv.NAME
:type: str
:value: >
   'spctsp'

```{autodoc2-docstring} src.envs.routing.spctsp.SPCTSPEnv.NAME
```

````

````{py:attribute} name
:canonical: src.envs.routing.spctsp.SPCTSPEnv.name
:type: str
:value: >
   'spctsp'

```{autodoc2-docstring} src.envs.routing.spctsp.SPCTSPEnv.name
```

````

````{py:attribute} _stochastic
:canonical: src.envs.routing.spctsp.SPCTSPEnv._stochastic
:type: bool
:value: >
   True

```{autodoc2-docstring} src.envs.routing.spctsp.SPCTSPEnv._stochastic
```

````

`````
