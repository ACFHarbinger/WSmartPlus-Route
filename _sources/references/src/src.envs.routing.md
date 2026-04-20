# {py:mod}`src.envs.routing`

```{py:module} src.envs.routing
```

```{autodoc2-docstring} src.envs.routing
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.envs.routing.cwcvrp
src.envs.routing.thop
src.envs.routing.atsp
src.envs.routing.pctsp
src.envs.routing.swcvrp
src.envs.routing.tsp
src.envs.routing.vrpp
src.envs.routing.cvrpp
src.envs.routing.wcvrp
src.envs.routing.irp
src.envs.routing.spctsp
src.envs.routing.op
src.envs.routing.cvrp
src.envs.routing.pdp
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_env <src.envs.routing.get_env>`
  - ```{autodoc2-docstring} src.envs.routing.get_env
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ENV_REGISTRY <src.envs.routing.ENV_REGISTRY>`
  - ```{autodoc2-docstring} src.envs.routing.ENV_REGISTRY
    :summary:
    ```
* - {py:obj}`__all__ <src.envs.routing.__all__>`
  - ```{autodoc2-docstring} src.envs.routing.__all__
    :summary:
    ```
````

### API

````{py:data} ENV_REGISTRY
:canonical: src.envs.routing.ENV_REGISTRY
:value: >
   None

```{autodoc2-docstring} src.envs.routing.ENV_REGISTRY
```

````

````{py:function} get_env(name: str, **kwargs) -> logic.src.envs.base.base.RL4COEnvBase
:canonical: src.envs.routing.get_env

```{autodoc2-docstring} src.envs.routing.get_env
```
````

````{py:data} __all__
:canonical: src.envs.routing.__all__
:value: >
   ['RL4COEnvBase', 'ImprovementEnvBase', 'Generator', 'VRPPGenerator', 'WCVRPGenerator', 'IRPGenerator...

```{autodoc2-docstring} src.envs.routing.__all__
```

````
