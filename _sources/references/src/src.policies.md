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
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.policies.branch_cut_and_price
src.policies.multi_vehicle
src.policies.vrpp_optimizer
src.policies.lin_kernighan
src.policies.last_minute
src.policies.hybrid_genetic_search
src.policies.adapters
src.policies.neural_agent
src.policies.adaptive_large_neighborhood_search
src.policies.regular
src.policies.policy_vrpp
src.policies.single_vehicle
src.policies.look_ahead
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
* - {py:obj}`RegularPolicyAdapter <src.policies.RegularPolicyAdapter>`
  - ```{autodoc2-docstring} src.policies.RegularPolicyAdapter
    :summary:
    ```
* - {py:obj}`LookAheadPolicyAdapter <src.policies.LookAheadPolicyAdapter>`
  - ```{autodoc2-docstring} src.policies.LookAheadPolicyAdapter
    :summary:
    ```
* - {py:obj}`LastMinutePolicyAdapter <src.policies.LastMinutePolicyAdapter>`
  - ```{autodoc2-docstring} src.policies.LastMinutePolicyAdapter
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
* - {py:obj}`ProfitPolicyAdapter <src.policies.ProfitPolicyAdapter>`
  - ```{autodoc2-docstring} src.policies.ProfitPolicyAdapter
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

````{py:data} RegularPolicyAdapter
:canonical: src.policies.RegularPolicyAdapter
:value: >
   None

```{autodoc2-docstring} src.policies.RegularPolicyAdapter
```

````

````{py:data} LookAheadPolicyAdapter
:canonical: src.policies.LookAheadPolicyAdapter
:value: >
   None

```{autodoc2-docstring} src.policies.LookAheadPolicyAdapter
```

````

````{py:data} LastMinutePolicyAdapter
:canonical: src.policies.LastMinutePolicyAdapter
:value: >
   None

```{autodoc2-docstring} src.policies.LastMinutePolicyAdapter
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

````{py:data} ProfitPolicyAdapter
:canonical: src.policies.ProfitPolicyAdapter
:value: >
   None

```{autodoc2-docstring} src.policies.ProfitPolicyAdapter
```

````

````{py:data} create_policy
:canonical: src.policies.create_policy
:value: >
   None

```{autodoc2-docstring} src.policies.create_policy
```

````
