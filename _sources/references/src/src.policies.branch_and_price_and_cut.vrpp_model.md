# {py:mod}`src.policies.branch_and_price_and_cut.vrpp_model`

```{py:module} src.policies.branch_and_price_and_cut.vrpp_model
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.vrpp_model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VRPPModel <src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel
    :summary:
    ```
````

### API

`````{py:class} VRPPModel(n_nodes: int, cost_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, num_vehicles: int = 1, revenue_per_kg: float = 1.0, cost_per_km: float = 1.0, mandatory_nodes: typing.Optional[typing.Set[int]] = None, node_coords: typing.Optional[typing.Union[numpy.ndarray, typing.Dict[int, typing.Tuple[float, float]]]] = None)
:canonical: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.__init__
```

````{py:method} get_edge_cost(i: int, j: int) -> float
:canonical: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.get_edge_cost

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.get_edge_cost
```

````

````{py:method} get_node_profit(i: int) -> float
:canonical: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.get_node_profit

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.get_node_profit
```

````

````{py:method} get_node_demand(i: int) -> float
:canonical: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.get_node_demand

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.get_node_demand
```

````

````{py:method} delta(node_set: typing.Set[int]) -> typing.List[typing.Tuple[int, int]]
:canonical: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.delta

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.delta
```

````

````{py:method} edges_in_set(node_set: typing.Set[int]) -> typing.List[typing.Tuple[int, int]]
:canonical: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.edges_in_set

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.edges_in_set
```

````

````{py:method} total_demand(node_set: typing.Set[int]) -> float
:canonical: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.total_demand

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.total_demand
```

````

````{py:method} validate_tour(tour: typing.List[int]) -> typing.Tuple[bool, str]
:canonical: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.validate_tour

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.validate_tour
```

````

````{py:method} compute_tour_profit(tour: typing.List[int]) -> float
:canonical: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.compute_tour_profit

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.compute_tour_profit
```

````

````{py:method} compute_tour_cost(tour: typing.List[int]) -> float
:canonical: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.compute_tour_cost

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel.compute_tour_cost
```

````

`````
