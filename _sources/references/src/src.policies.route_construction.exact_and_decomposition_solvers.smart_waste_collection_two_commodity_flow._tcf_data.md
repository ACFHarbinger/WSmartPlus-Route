# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TCFData <src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`build_tcf_data <src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.build_tcf_data>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.build_tcf_data
    :summary:
    ```
````

### API

`````{py:class} TCFData
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData
```

````{py:attribute} Q
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.Q
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.Q
```

````

````{py:attribute} R
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.R
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.R
```

````

````{py:attribute} C
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.C
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.C
```

````

````{py:attribute} Omega
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.Omega
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.Omega
```

````

````{py:attribute} psi
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.psi
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.psi
```

````

````{py:attribute} n_bins
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.n_bins
:type: int
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.n_bins
```

````

````{py:attribute} nodes
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.nodes
:type: typing.List[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.nodes
```

````

````{py:attribute} nodes_real
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.nodes_real
:type: typing.List[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.nodes_real
```

````

````{py:attribute} S_dict
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.S_dict
:type: typing.Dict[int, float]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.S_dict
```

````

````{py:attribute} pure_binsids
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.pure_binsids
:type: typing.List[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.pure_binsids
```

````

````{py:attribute} criticos_dict
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.criticos_dict
:type: typing.Dict[int, bool]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.criticos_dict
```

````

````{py:attribute} valid_arcs
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.valid_arcs
:type: typing.List[typing.Tuple[int, int]]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.valid_arcs
```

````

````{py:attribute} id_map
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.id_map
:type: typing.Dict[int, int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.id_map
```

````

````{py:attribute} max_trucks
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.max_trucks
:type: int
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData.max_trucks
```

````

`````

````{py:function} build_tcf_data(bins: numpy.typing.NDArray[numpy.float64], distance_matrix: typing.List[typing.List[float]], values: typing.Dict[str, float], binsids: typing.List[int], mandatory: typing.List[int], number_vehicles: int) -> src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.build_tcf_data

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.build_tcf_data
```
````
