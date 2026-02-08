# {py:mod}`src.envs.generators`

```{py:module} src.envs.generators
```

```{autodoc2-docstring} src.envs.generators
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.envs.generators.vrpp
src.envs.generators.wcvrp
src.envs.generators.scwcvrp
src.envs.generators.tsp
src.envs.generators.base
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_generator <src.envs.generators.get_generator>`
  - ```{autodoc2-docstring} src.envs.generators.get_generator
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GENERATOR_REGISTRY <src.envs.generators.GENERATOR_REGISTRY>`
  - ```{autodoc2-docstring} src.envs.generators.GENERATOR_REGISTRY
    :summary:
    ```
* - {py:obj}`__all__ <src.envs.generators.__all__>`
  - ```{autodoc2-docstring} src.envs.generators.__all__
    :summary:
    ```
````

### API

````{py:data} GENERATOR_REGISTRY
:canonical: src.envs.generators.GENERATOR_REGISTRY
:type: dict[str, type[src.envs.generators.base.Generator]]
:value: >
   None

```{autodoc2-docstring} src.envs.generators.GENERATOR_REGISTRY
```

````

````{py:function} get_generator(name: str, **kwargs: typing.Any) -> src.envs.generators.base.Generator
:canonical: src.envs.generators.get_generator

```{autodoc2-docstring} src.envs.generators.get_generator
```
````

````{py:data} __all__
:canonical: src.envs.generators.__all__
:value: >
   ['Generator', 'VRPPGenerator', 'WCVRPGenerator', 'SCWCVRPGenerator', 'TSPGenerator', 'TSPGenerator',...

```{autodoc2-docstring} src.envs.generators.__all__
```

````
