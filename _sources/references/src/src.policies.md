# {py:mod}`src.policies`

```{py:module} src.policies
```

```{autodoc2-docstring} src.policies
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

src.policies.alns_aux
src.policies.look_ahead_aux
src.policies.hgs_aux
src.policies.selection
src.policies.adapters
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.policies.hgs_alns_solver
src.policies.base_routing_policy
src.policies.branch_cut_and_price
src.policies.multi_vehicle
src.policies.must_go_selection
src.policies.lin_kernighan
src.policies.hybrid_genetic_search
src.policies.neural_agent
src.policies.adaptive_large_neighborhood_search
src.policies.single_vehicle
src.policies.post_processing
```

## Package Contents

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.policies.__all__>`
  - ```{autodoc2-docstring} src.policies.__all__
    :summary:
    ```
* - {py:obj}`NeuralPolicyAdapter <src.policies.NeuralPolicyAdapter>`
  - ```{autodoc2-docstring} src.policies.NeuralPolicyAdapter
    :summary:
    ```
* - {py:obj}`VRPPPolicyAdapter <src.policies.VRPPPolicyAdapter>`
  - ```{autodoc2-docstring} src.policies.VRPPPolicyAdapter
    :summary:
    ```
* - {py:obj}`create_policy <src.policies.create_policy>`
  - ```{autodoc2-docstring} src.policies.create_policy
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: src.policies.__all__
:value: >
   ['ALNSParams', 'PolicyFactory', 'create_points', 'create_policy', 'find_route', 'find_routes', 'find...

```{autodoc2-docstring} src.policies.__all__
```

````

````{py:data} NeuralPolicyAdapter
:canonical: src.policies.NeuralPolicyAdapter
:value: >
   None

```{autodoc2-docstring} src.policies.NeuralPolicyAdapter
```

````

````{py:data} VRPPPolicyAdapter
:canonical: src.policies.VRPPPolicyAdapter
:value: >
   None

```{autodoc2-docstring} src.policies.VRPPPolicyAdapter
```

````

````{py:data} create_policy
:canonical: src.policies.create_policy
:value: >
   None

```{autodoc2-docstring} src.policies.create_policy
```

````
