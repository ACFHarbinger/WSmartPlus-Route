# {py:mod}`src.envs`

```{py:module} src.envs
```

```{autodoc2-docstring} src.envs
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

src.envs.generators
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.envs.cvrpp
src.envs.vrpp
src.envs.swcvrp
src.envs.wcvrp
src.envs.tasks
src.envs.cwcvrp
src.envs.problems
src.envs.sdwcvrp
src.envs.tsp_kopt
src.envs.tsp
src.envs.base
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_env <src.envs.get_env>`
  - ```{autodoc2-docstring} src.envs.get_env
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ENV_REGISTRY <src.envs.ENV_REGISTRY>`
  - ```{autodoc2-docstring} src.envs.ENV_REGISTRY
    :summary:
    ```
* - {py:obj}`__all__ <src.envs.__all__>`
  - ```{autodoc2-docstring} src.envs.__all__
    :summary:
    ```
````

### API

````{py:data} ENV_REGISTRY
:canonical: src.envs.ENV_REGISTRY
:value: >
   None

```{autodoc2-docstring} src.envs.ENV_REGISTRY
```

````

````{py:function} get_env(name: str, **kwargs) -> logic.src.envs.base.RL4COEnvBase
:canonical: src.envs.get_env

```{autodoc2-docstring} src.envs.get_env
```
````

````{py:data} __all__
:canonical: src.envs.__all__
:value: >
   ['RL4COEnvBase', 'ImprovementEnvBase', 'Generator', 'VRPPGenerator', 'WCVRPGenerator', 'get_generato...

```{autodoc2-docstring} src.envs.__all__
```

````
