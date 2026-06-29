# {py:mod}`src.policies.vector`

```{py:module} src.policies.vector
```

```{autodoc2-docstring} src.policies.vector
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

src.policies.vector.selection
src.policies.vector.shared
src.policies.vector.hgs_core
src.policies.vector.operators
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.policies.vector.ant_colony_system
src.policies.vector.hgs
src.policies.vector.iterated_local_search
src.policies.vector.alns
src.policies.vector.adaptive_large_neighborhood_search
src.policies.vector.hgs_alns
src.policies.vector.random_local_search
src.policies.vector.hybrid_volleyball_premier_league
src.policies.vector.local_search
src.policies.vector.hybrid_genetic_search
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_policy_class <src.policies.vector.get_policy_class>`
  - ```{autodoc2-docstring} src.policies.vector.get_policy_class
    :summary:
    ```
* - {py:obj}`get_policy <src.policies.vector.get_policy>`
  - ```{autodoc2-docstring} src.policies.vector.get_policy
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_POLICY_REGISTRY_SPEC <src.policies.vector._POLICY_REGISTRY_SPEC>`
  - ```{autodoc2-docstring} src.policies.vector._POLICY_REGISTRY_SPEC
    :summary:
    ```
* - {py:obj}`__all__ <src.policies.vector.__all__>`
  - ```{autodoc2-docstring} src.policies.vector.__all__
    :summary:
    ```
````

### API

````{py:data} _POLICY_REGISTRY_SPEC
:canonical: src.policies.vector._POLICY_REGISTRY_SPEC
:type: typing.Dict[str, typing.Tuple[str, str]]
:value: >
   None

```{autodoc2-docstring} src.policies.vector._POLICY_REGISTRY_SPEC
```

````

````{py:function} get_policy_class(name: str) -> type
:canonical: src.policies.vector.get_policy_class

```{autodoc2-docstring} src.policies.vector.get_policy_class
```
````

````{py:function} get_policy(name: str, **kwargs: typing.Any) -> torch.nn.Module
:canonical: src.policies.vector.get_policy

```{autodoc2-docstring} src.policies.vector.get_policy
```
````

````{py:data} __all__
:canonical: src.policies.vector.__all__
:value: >
   ['ConstructivePolicy', 'ImprovementPolicy', 'AttentionModelPolicy', 'DeepDecoderPolicy', 'DeepACOPol...

```{autodoc2-docstring} src.policies.vector.__all__
```

````
