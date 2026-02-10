# {py:mod}`src.models.policies`

```{py:module} src.models.policies
```

```{autodoc2-docstring} src.models.policies
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

src.models.policies.operators
src.models.policies.hgs
src.models.policies.selection
src.models.policies.shared
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.models.policies.random_local_search
src.models.policies.hybrid_genetic_search
src.models.policies.adaptive_large_neighborhood_search
src.models.policies.alns
src.models.policies.local_search
src.models.policies.hgs_alns
src.models.policies.iterated_local_search
src.models.policies.ant_colony_system
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_policy_class <src.models.policies.get_policy_class>`
  - ```{autodoc2-docstring} src.models.policies.get_policy_class
    :summary:
    ```
* - {py:obj}`get_policy <src.models.policies.get_policy>`
  - ```{autodoc2-docstring} src.models.policies.get_policy
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_POLICY_REGISTRY_SPEC <src.models.policies._POLICY_REGISTRY_SPEC>`
  - ```{autodoc2-docstring} src.models.policies._POLICY_REGISTRY_SPEC
    :summary:
    ```
* - {py:obj}`__all__ <src.models.policies.__all__>`
  - ```{autodoc2-docstring} src.models.policies.__all__
    :summary:
    ```
````

### API

````{py:data} _POLICY_REGISTRY_SPEC
:canonical: src.models.policies._POLICY_REGISTRY_SPEC
:value: >
   None

```{autodoc2-docstring} src.models.policies._POLICY_REGISTRY_SPEC
```

````

````{py:function} get_policy_class(name: str) -> type
:canonical: src.models.policies.get_policy_class

```{autodoc2-docstring} src.models.policies.get_policy_class
```
````

````{py:function} get_policy(name: str, **kwargs) -> torch.nn.Module
:canonical: src.models.policies.get_policy

```{autodoc2-docstring} src.models.policies.get_policy
```
````

````{py:data} __all__
:canonical: src.models.policies.__all__
:value: >
   ['ConstructivePolicy', 'ImprovementPolicy', 'AttentionModelPolicy', 'DeepDecoderPolicy', 'DeepACOPol...

```{autodoc2-docstring} src.models.policies.__all__
```

````
